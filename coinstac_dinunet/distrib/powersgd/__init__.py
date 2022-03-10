import os as _os

import numpy as _np
import torch as _torch
from coinstac_dinunet import config as _conf
from coinstac_dinunet.utils import tensorutils as _tu

from ..learner import COINNLearner as _COINNLearner
from ..reducer import COINNReducer as _COINNReducer
from collections import OrderedDict as _Dict

_sep = _os.sep


def _orthogonalize(matrix, epsilon=1e-8):
    """
    Reference: torch.com: torch.distributed.algorithms.commn_hook.powerGSD_hooks"""
    """
    Applies Gram-Schmidt procedure to orthogonalize a given 2D tensor.
    If epsilon is 0, this is equivalent to `torch.qr(matrix, out=(matrix, _))`,
    but `torch.qr` is very slow, probably because it is not optimized for a matrix that has a small number of columns.
    """
    num_cols = matrix.shape[1]
    for i in range(num_cols):
        # Normalize the i'th column.
        col = matrix[:, i: i + 1]
        # If no epsilon is added here, division by zero may be caused by vanishing gradients.
        # This epsilon is not needed if the input matrix covers the gradients of at least one entire layer in the neural network.
        if epsilon == 0:
            # Note that col ** 2 can underflow/overflow if we use FP16.
            # May need to consider multiplying a scaling factor and dividing it later, or using bfloat16 instead.
            col /= _torch.norm(col)
        else:
            col /= _torch.norm(col) + epsilon
        # Project it on the rest and remove it.
        if i + 1 < num_cols:
            rest = matrix[:, i + 1:]
            rest -= _torch.sum(col * rest, dim=0) * col


class PowerSGDState:
    def __init__(self):
        self.error_dict = _Dict()
        self.p_memory_dict = _Dict()
        self.q_memory_dict = _Dict()
        self.rank1_tensors = _Dict()
        self.high_rank_tensors = _Dict()
        self.iter = 0


class PowerSGDLearner(_COINNLearner):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.matrix_approximation_rank = self.cache.setdefault('matrix_approximation_rank', 1)
        self.start_powerSGD_iter = self.cache.setdefault('start_powerSGD_iter', 10)
        self.use_error_feedback = self.cache.setdefault('use_error_feedback', True)
        self.warm_start = self.cache.setdefault('warm_start', True)
        self.seed = self.cache.get('seed')
        self.powerSGD_state = self.cache.setdefault('powerSGD_state', PowerSGDState())

    def step(self) -> dict:
        if self.powerSGD_state.iter < self.start_powerSGD_iter:
            self.powerSGD_state.iter += 1
            return super(PowerSGDLearner, self).step()

        out = {}
        file_Qs = self.state['baseDirectory'] + _os.sep + self.input['powerSGD_Q_file_AGG']
        received_Qs = _tu.load_arrays(file_Qs)
        for k, new_q in zip(self.powerSGD_state.p_memory_dict, received_Qs):
            self.powerSGD_state.q_memory_dict[k] = _torch.from_numpy(new_q).to(self.device)

        file_rank_1 = self.state['baseDirectory'] + _os.sep + self.input['rank_1_grads_file_AGG']
        rank_1_grads = [_torch.from_numpy(t).to(self.device).float()
                        for t in _tu.load_arrays(file_rank_1)][::-1]

        first_model = list(self.trainer.nn.keys())[0]
        for param_key, param in self.trainer.nn[first_model].named_parameters():
            if param.grad.ndimension() <= 1:
                param.grad = rank_1_grads.pop()
            else:
                new_grad = _torch.matmul(
                    self.powerSGD_state.p_memory_dict[param_key],
                    self.powerSGD_state.q_memory_dict[param_key].t()).float()
                if self.use_error_feedback:
                    self.powerSGD_state.error_dict[param_key] = _torch.clone(param.grad).detach() - new_grad
                param.grad = new_grad

        first_optim = list(self.trainer.optimizer.keys())[0]
        self.trainer.optimizer[first_optim].step()
        self.powerSGD_state.iter += 1
        return out

    def _prepare_parameters(self):
        it, out = self.backward()

        first_model = list(self.trainer.nn.keys())[0]
        for key, param in self.trainer.nn[first_model].named_parameters():
            if param.ndimension() <= 1:
                self.powerSGD_state.rank1_tensors[key] = _torch.clone(param.grad).detach()
            else:
                self.powerSGD_state.high_rank_tensors[key] = _torch.clone(param.grad).detach()

        for param_key, M in self.powerSGD_state.high_rank_tensors.items():
            if self.use_error_feedback:
                if param_key in self.powerSGD_state.error_dict:
                    M += self.powerSGD_state.error_dict[param_key]
                else:
                    self.powerSGD_state.error_dict[param_key] = _torch.zeros(M.shape, device=self.device, dtype=M.dtype)

            need_randomize_qs = not self.warm_start or param_key not in self.powerSGD_state.p_memory_dict
            if need_randomize_qs:
                n, m = M.shape
                _torch.manual_seed(self.seed + self.powerSGD_state.iter)
                self.powerSGD_state.q_memory_dict[param_key] = _torch.randn(
                    (m, self.matrix_approximation_rank),
                    device=self.device,
                    dtype=M.dtype,
                )
                _orthogonalize(self.powerSGD_state.q_memory_dict[param_key])
            else:
                _orthogonalize(self.powerSGD_state.q_memory_dict[param_key])

            self.powerSGD_state.p_memory_dict[param_key] = _torch.matmul(
                M,
                self.powerSGD_state.q_memory_dict[param_key]
            )

        return it, out

    def to_reduce(self):
        if self.powerSGD_state.iter < self.start_powerSGD_iter:
            it, out = super(PowerSGDLearner, self).to_reduce()
            out['start_power_iter'] = False
            return it, out

        it, out = {}, {}
        if self.input.get('powerSGD_phase', 'phase_P_sync') == 'phase_P_sync':
            it, out = self._prepare_parameters()
            Ps = [p.clone().detach().cpu().numpy().astype(self.dtype)
                  for p in self.powerSGD_state.p_memory_dict.values()]
            out['powerSGD_P_file'] = f"powerSGD_P_{_conf.grads_file}"
            _tu.save_arrays(
                self.state['transferDirectory'] + _sep + out['powerSGD_P_file'],
                _np.array(Ps, dtype=object)
            )

        elif self.input.get('powerSGD_phase') == 'phase_Q_sync':
            out['rank1_grads_file'] = f"rank1_{_conf.grads_file}"
            rank1_grads = [g.cpu().numpy().astype(self.dtype)
                           for g in self.powerSGD_state.rank1_tensors.values()]
            _tu.save_arrays(
                self.state['transferDirectory'] + _sep + out['rank1_grads_file'],
                _np.array(rank1_grads, dtype=object)
            )
            self.powerSGD_state.rank1_tensors = {}

            """Recev all Ps"""
            file_Ps = self.state['baseDirectory'] + _os.sep + self.input['powerSGD_P_file_AGG']
            received_Ps = _tu.load_arrays(file_Ps)
            for k, new_p in zip(self.powerSGD_state.p_memory_dict, received_Ps):
                self.powerSGD_state.p_memory_dict[k] = _torch.from_numpy(new_p).to(self.device)
                _orthogonalize(self.powerSGD_state.p_memory_dict[k])

            """Send all Qs"""
            for param_key, M in self.powerSGD_state.high_rank_tensors.items():
                self.powerSGD_state.q_memory_dict[param_key] = _torch.matmul(
                    M.t(),
                    self.powerSGD_state.p_memory_dict[param_key]
                )
            Qs = [p.cpu().numpy().astype(self.dtype)
                  for p in self.powerSGD_state.q_memory_dict.values()]
            out['powerSGD_Q_file'] = f"powerSGD_Q_{_conf.grads_file}"
            _tu.save_arrays(
                self.state['transferDirectory'] + _sep + out['powerSGD_Q_file'],
                _np.array(Qs, dtype=object)
            )
            self.powerSGD_state.high_rank_tensors = {}

        out['start_power_iter'] = True
        out['reduce'] = True
        return it, out


class PowerSGDReducer(_COINNReducer):

    def reduce(self):
        """ Average each sites gradients and pass it to all sites. """
        site = list(self.input.values())[0]
        if not site['start_power_iter']:
            return super(PowerSGDReducer, self).reduce()

        out = {}
        if site.get('powerSGD_P_file'):
            """Average and orthogonalize Ps"""
            out['powerSGD_P_file_AGG'] = f"powerSGD_P_{_conf.avg_grads_file}"
            _tu.save_arrays(
                self.state['transferDirectory'] + _os.sep + out['powerSGD_P_file_AGG'],
                _np.array(self._average('powerSGD_P_file'), dtype=object)
            )
            out['powerSGD_phase'] = 'phase_Q_sync'

        elif site.get('powerSGD_Q_file'):
            """Average Qs and rank1_grads_file"""

            out['rank_1_grads_file_AGG'] = f"rank1_AGG_{_conf.avg_grads_file}"
            _tu.save_arrays(
                self.state['transferDirectory'] + _os.sep + out['rank_1_grads_file_AGG'],
                _np.array(self._average('rank1_grads_file'), dtype=object)
            )

            out['powerSGD_Q_file_AGG'] = f"powerSGD_Q_{_conf.avg_grads_file}"
            _tu.save_arrays(
                self.state['transferDirectory'] + _os.sep + out['powerSGD_Q_file_AGG'],
                _np.array(self._average('powerSGD_Q_file'), dtype=object)
            )
            out['powerSGD_phase'] = 'phase_P_sync'
            out['update'] = True

        return out
