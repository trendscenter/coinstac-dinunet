import os as _os
import torch as _torch
import numpy as _np
from coinstac_dinunet.utils import tensorutils as _tu

_SKIP_NORM_Layers = [_torch.nn.BatchNorm1d, _torch.nn.LayerNorm, _torch.nn.GroupNorm]


def power_iteration_BC(B, C, rank=10, numiterations=20, device='cuda', tol=1e-3):
    [cm, cn] = C.shape
    if cm > cn:
        CC = _torch.mm(C.T, C)
        BCC = _torch.mm(B, CC)
    else:
        BCT = _torch.mm(B, C.T)
        BCC = _torch.mm(BCT, BCT.T)

    def zero_result():
        sigma = _torch.tensor(0.0, device=device)
        b_k = _torch.zeros(B.shape[0], device=device)
        c_k = _torch.zeros(C.shape[0], device=device)
        return {"b": b_k, "c": c_k, "sigma": sigma, "v": b_k}

    def eigenvalue(B, v):
        Bv = _torch.mv(B.T, v)
        return _torch.sqrt(Bv.dot(_torch.mv(CC, Bv)))

    def eigenvalue2(B, v):
        Bv = _torch.mv(_torch.mm(C, B.T), v)
        return _torch.sqrt(Bv.T.dot(Bv))

    def past_values(computed_eigs):
        bb = _torch.stack([x['b'] for x in computed_eigs], 0)
        vv = _torch.stack([x['v'] for x in computed_eigs], 0)
        return bb, vv

    def iterations(computed_eigs=[], is_sigma=1):
        if not is_sigma: return zero_result()
        # start with one of the columns
        b_k = _torch.rand(B.shape[0], device=device)
        # b_k = B[:, 0]  # np.random.randn(B.shape[0])
        if computed_eigs:
            bb, vv = past_values(computed_eigs)
        for _ in range(numiterations):
            adjuster = _torch.tensor(0.0, device=device)
            if computed_eigs:
                adjuster = _torch.mv(vv.T, _torch.mv(bb, b_k))
            # calculate the matrix-by-vector product (BC'CB' - adjusting_matrix)b
            if cm > cn:
                b_k1 = _torch.mv(BCC, _torch.mv(B.T, b_k)) - adjuster
            else:
                b_k1 = _torch.mv(BCC, b_k) - adjuster
            # calculate the norm of b
            b_k1_norm = _torch.norm(b_k1)
            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        if cm > cn:
            sigma = eigenvalue(B, b_k)
        else:
            sigma = eigenvalue2(B, b_k)
        if _torch.isnan(sigma): return zero_result()
        if cm > cn:
            c_k = _torch.mv(C, _torch.mv(B.T, b_k)) / sigma
        else:
            c_k = _torch.mv(BCT.T, b_k) / sigma
        if len(computed_eigs) > 1 and _torch.norm(b_k - computed_eigs[-1]['b']) / _torch.norm(
                computed_eigs[-1]['b']) < tol:
            r = zero_result()
            computed_eigs[-1]['b'] = r['b']
            computed_eigs[-1]['c'] = r['c']
            computed_eigs[-1]['sigma'] = r['sigma']
            return zero_result()
        return {"b": b_k, "c": c_k, "sigma": sigma, "v": sigma * sigma * b_k}

    eigs = [{"sigma": _torch.tensor(1.0, device=device)}]
    for i in range(rank):
        eigs += [iterations(computed_eigs=eigs[1:], is_sigma=eigs[-1]["sigma"])]
        if eigs[-1]["sigma"] == 0.0:
            break
    eigs = eigs[1:-1]
    return (
        _torch.stack([x["sigma"] * x["b"] for x in eigs], 1),
        _torch.stack([x["c"] for x in eigs], 1),)


def _dad_trainable_module(module):
    a_norm_layer = any([isinstance(module, k) for k in _SKIP_NORM_Layers])
    if a_norm_layer:
        return False

    """Has trainable parameters"""
    return len(list(module.parameters())) > 0


class DADParallel(_torch.nn.Module):
    def __init__(self, module, cache=None, input=None, state=None, device=None, dtype='float32', **kw):
        super().__init__()
        self.module = module.module if isinstance(module, DADParallel) else module
        self.cache = cache
        self.input = input
        self.state = state
        self.device = device
        self.dtype = dtype
        self._is_dad_module = self.cache.setdefault('is_dad_module', {})
        self._reset()

    def _reset(self):
        self.fw_hooks_handle = []
        self.bk_hooks_handle = []
        self._activations = {}
        self._local_grads = {}

    def _hook_fn(self, hook_type, hook_key):
        def get(m, in_grad, out_grad):
            if hook_type.lower() == 'forward':
                for i, b in enumerate(in_grad):
                    if b is not None:
                        self._activations[hook_key] = b
                    break
            if hook_type.lower() == 'backward':
                for i, c in enumerate(out_grad):
                    if c is not None:
                        self._local_grads[hook_key] = c
                    break

        return get

    def _hook(self):
        def _hook_recursive(module_name, module):
            children = list(module.named_children())
            self._is_dad_module[module_name] = False
            if len(children) > 0:
                for children_name, child in children:
                    _hook_recursive(
                        self._hierarchy_key(module_name, children_name),
                        child
                    )

            elif _dad_trainable_module(module):
                self._is_dad_module[module_name] = True
                self.fw_hooks_handle.append(
                    module.register_forward_hook(self._hook_fn('forward', module_name))
                )
                self.bk_hooks_handle.append(
                    module.register_backward_hook(self._hook_fn('backward', module_name))
                )

        if self.training:
            for ch_name, ch in list(self.module.named_children()):
                _hook_recursive(ch_name, ch)

    def _unhook(self):
        for hk in self.fw_hooks_handle:
            hk.remove()
        for hk in self.bk_hooks_handle:
            hk.remove()

    def train(self, mode=True):
        self.module.train(mode)
        self._hook()
        return self

    def eval(self):
        self.module.eval()
        self._unhook()
        return self

    def forward(self, *inputs, **kwargs):
        if self.training:
            self._reset()
        output = self.module(*inputs, **kwargs)
        return output

    def _hierarchy_key(self, *args):
        return ".".join([f"{a}" for a in args])

    def synced_param_update(self):
        def _sync(module_name, module, data):
            dad_params = dict(list(module.named_parameters())[::-1])
            dad_children = dict(list(module.named_children())[::-1])
            if len(dad_children) > 0:
                for child_name, child in dad_children.items():
                    _sync(self._hierarchy_key(module_name, child_name), child, data)

            elif self._is_dad_module.get(module_name):
                act_tall, local_grad_tall = data.pop()

                act_tall = _torch.tensor(act_tall, dtype=_torch.float32)
                local_grad_tall = _torch.tensor(local_grad_tall.squeeze(), dtype=_torch.float32)

                dad_params["weight"].grad.data = (act_tall.T.mm(local_grad_tall)).T.contiguous()
                if dad_params.get("bias") is not None:
                    dad_params[f"bias"].grad.data = local_grad_tall.sum(0)

        _data = _tu.load_arrays(self.state['baseDirectory'] + _os.sep + self.input['reduced_dad_data']).tolist()[::-1]
        for ch_name, ch in list(self.module.named_children())[::-1]:
            _sync(ch_name, ch, _data)

    def dad_backward(self):

        def _backward(module_name, module, data):
            dad_children = dict(list(module.named_children())[::-1])

            if len(dad_children) > 0:
                for child_name, child in dad_children.items():
                    _backward(self._hierarchy_key(module_name, child_name), child, data)

            elif self._is_dad_module.get(module_name):
                delta, act = power_iteration_BC(
                    self._local_grads[module_name].T,
                    self._activations[module_name].T,
                    rank=self.cache.setdefault('dad_reduction_rank', 10),
                    numiterations=self.cache.setdefault('dad_num_pow_iters', 5),
                    device=self.device,
                    tol=self.cache.setdefault('dad_tol', 1e-3)
                )
                data.append([act.T.detach().cpu().numpy().astype(self.dtype),
                             delta.T.detach().cpu().numpy().astype(self.dtype)[None, ...]])

        out, data = {}, []
        for ch_name, ch in list(self.module.named_children())[::-1]:
            _backward(ch_name, ch, data)

        out['dad_data'] = 'dad_data.npy'
        _tu.save_arrays(
            self.state['transferDirectory'] + _os.sep + out['dad_data'],
            _np.array(data, dtype=object)
        )
        return out
