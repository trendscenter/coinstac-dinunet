import logging
import math

import numpy as np
import torch
import torch.distributed as dist

from . import default_hooks as default

import torch.distributed as dist
dist._GradBucket
def _orthogonalize(matrix, epsilon=1e-8):
    """
    Applies Gram-Schmidt procedure to orthogonalize a given 2D tensor.
    If epsilon is 0, this is equivalent to `torch.qr(matrix, out=(matrix, _))`,
    but `torch.qr` is very slow, probably because it is not optimized for a matrix that has a small number of columns.
    """
    num_cols = matrix.shape[1]
    for i in range(num_cols):
        # Normalize the i'th column.
        col = matrix[:, i : i + 1]
        # If no epsilon is added here, division by zero may be caused by vanishing gradients.
        # This epsilon is not needed if the input matrix covers the gradients of at least one entire layer in the neural network.
        if epsilon == 0:
            # Note that col ** 2 can underflow/overflow if we use FP16.
            # May need to consider multiplying a scaling factor and dividing it later, or using bfloat16 instead.
            col /= torch.norm(col)
        else:
            col /= torch.norm(col) + epsilon
        # Project it on the rest and remove it.
        if i + 1 < num_cols:
            rest = matrix[:, i + 1 :]
            rest -= torch.sum(col * rest, dim=0) * col

def powerSGD_hook(state, bucket) -> torch.futures.Future:
    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.get_tensors()[0]

    # Run vanilla allreduce in the first `start_powerSGD_iter` iterations.
    if state.iter < state.start_powerSGD_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(group_to_use, input_tensor)

    # Apply PowerSGD after `start_powerSGD_iter` iterations.
    device = input_tensor.device
    dtype = input_tensor.dtype

    # Incorporate the error from the previous state into the gradients.
    bucket_index = bucket.get_index()
    input_tensor_cp = None
    total_length = input_tensor.shape[0]
    if state.use_error_feedback:
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            logging.info(
                "A zero tensor of length {} that represents local error is created.".format(
                    total_length
                )
            )
            state.error_dict[bucket_index] = torch.zeros(
                total_length, device=device, dtype=dtype
            )

        # Keep a copy of the input tensor,
        # so that we can compute the local error caused by compression later,
        # by comparing this copy and the input tensor updated after decompression.
        input_tensor_cp = torch.clone(input_tensor).detach()

    # Unflatten the input tensor into per-parameter tensors, for layer-wise compression.
    tensors = [
        input_tensor[offset : offset + length].view(sizes)
        for offset, length, sizes in zip(
            bucket.get_offsets(), bucket.get_lengths(), bucket.get_sizes_list()
        )
    ]

    # Step I: Handle rank-1 tensors.
    # Allocate contiguous memory for rank-1 tensors to allreduce them without compression efficiently.
    rank1_tensors = [tensor for tensor in tensors if tensor.ndimension() <= 1]
    rank1_tensors_memory = (
        torch.cat([tensor.view(-1) for tensor in rank1_tensors])
        if rank1_tensors
        else torch.tensor([], device=device, dtype=dtype)
    )

    # Step II: Handle high-rank tensors.
    # Allocate contiguous memory for Ps and Qs to allreduce compressed high-rank tensors efficiently.
    high_rank_tensors = [
        tensor.view(tensor.shape[0], -1)
        for tensor in tensors
        if tensor.ndimension() > 1
    ]
    total_Ps_size = 0
    total_Qs_size = 0
    for tensor in high_rank_tensors:
        n, m = tensor.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        total_Ps_size += n * matrix_approximation_rank
        total_Qs_size += m * matrix_approximation_rank
    # If warm-start is enabled, reuse Ps and Qs from the previous iteration if possible.
    # The memory spaces of Ps and Qs need to be allocated in the first iteration when PowerSGD is applied.
    need_randomize_qs = False
    if not state.warm_start or bucket_index not in state.p_memory_dict:
        need_randomize_qs = True
        # If warm-start is disabled, low-rank tensors will be initialized at every step.
        # Only log this if warm-start to avoid spamming.
        if state.warm_start:
            logging.info(
                "Allocating contiguous memory of length {} for Ps, and of length {} for Qs, respectively.".format(
                    total_Ps_size, total_Qs_size
                )
            )
        state.p_memory_dict[bucket_index] = torch.empty(
            total_Ps_size, device=device, dtype=dtype
        )
        state.q_memory_dict[bucket_index] = torch.empty(
            total_Qs_size, device=device, dtype=dtype
        )

    # Create Ps and Qs that point to the allocated memory.
    ps = []
    qs = []
    p_idx = 0
    q_idx = 0
    for tensor in high_rank_tensors:
        n, m = tensor.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        ps.append(
            state.p_memory_dict[bucket_index][
                p_idx : p_idx + n * matrix_approximation_rank
            ].view(n, matrix_approximation_rank)
        )
        qs.append(
            state.q_memory_dict[bucket_index][
                q_idx : q_idx + m * matrix_approximation_rank
            ].view(m, matrix_approximation_rank)
        )
        p_idx += n * matrix_approximation_rank
        q_idx += m * matrix_approximation_rank

    # If warm-start is enabled, reuse Qs from the previous iteration if possible and skip filling random values.
    # The exception is the first iteration when PowerSGD is applied.
    if not need_randomize_qs:
        for q in qs:
            _orthogonalize(q)
    else:
        with torch.random.fork_rng(devices=[]):
            # Fork this RNG to avoid changing the seed globally and affecting the random sampling anywhere else in the training.
            # The seed makes sure that the initial random values are the same across all the DDP replicas.
            # This seed should differ at every step.
            # Since it is very slow to fork RNG state across all the CUDA devices,
            # only fork on CPU and then move the generated tensor to the CUDA device (by overwriting q).
            torch.manual_seed(state.rng.randint(1_000_000_000))
            for q in qs:
                q.copy_(
                    torch.randn(
                        *q.shape,
                        device="cpu",
                        dtype=dtype,
                    )
                )
                _orthogonalize(q)

    # Compute Ps.
    for tensor, q, p in zip(high_rank_tensors, qs, ps):
        torch.matmul(tensor, q, out=p)

    # This allreduce is only applied to rank-1 tensors,
    # so it should have been kicked off before the above computation on the high-rank tensors to hide more communication costs.
    # However, this somehow requires a separate future chain at this time.
    allreduce_contiguous_rank1_tensors_fut = dist.all_reduce(
        rank1_tensors_memory, group=group_to_use, async_op=True
    ).get_future()

    def unpack_rank1_tensors_and_allreduce_ps(fut):
        rank1_tensors_memory = fut.value()[0].div_(world_size)
        idx = 0
        for tensor in rank1_tensors:
            tensor.copy_(rank1_tensors_memory[idx : idx + tensor.shape[0]])
            idx += tensor.shape[0]

        # Since these Ps will be orthogonalized later, no need to divide them by world size.
        return [
            dist.all_reduce(
                state.p_memory_dict[bucket_index], group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]
        ]

    def compute_qs(fut):
        state.p_memory_dict[bucket_index] = fut.value()[0]
        for p in ps:
            _orthogonalize(p)

        # Compute Qs.
        for tensor, p, q in zip(high_rank_tensors, ps, qs):
            torch.matmul(tensor.t(), p, out=q)

        # TODO: The above procedure does two matmul+allreduce steps per iteration --
        # one left multiplication and one right multiplication.
        # For warm-start, can take one such step at a time, and alternate between them.

        # Allreduce Qs.
        return [
            dist.all_reduce(
                state.q_memory_dict[bucket_index], group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]
        ]

    def decompress(fut):
        state.q_memory_dict[bucket_index] = fut.value()[0].div_(world_size)

        for p, q, tensor in zip(ps, qs, high_rank_tensors):
            torch.matmul(p, q.t(), out=tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        if state.use_error_feedback:
            # Memorize the local errors.
            state.error_dict[bucket_index] = input_tensor_cp - input_tensor
        if not state.warm_start:
            state.p_memory_dict.clear()
            state.q_memory_dict.clear()

        state.maybe_increase_iter(bucket)

        return [input_tensor]

    return (
        allreduce_contiguous_rank1_tensors_fut.then(
            unpack_rank1_tensors_and_allreduce_ps
        )
        .then(compute_qs)
        .then(decompress)
    )
