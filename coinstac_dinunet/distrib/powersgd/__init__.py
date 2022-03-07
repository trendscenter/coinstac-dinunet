import os as _os

import numpy as _np
import torch

from coinstac_dinunet import config as _conf
from coinstac_dinunet.utils import tensorutils as _tu
from functools import partial as _partial

from .. import COINNLearner as _COINNLearner
from .. import COINNReducer as _COINNReducer
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import _orthogonalize

_sep = _os.sep


class PowerSGDLearner(_COINNLearner):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.matrix_approximation_rank = self.cache.setdefault('matrix_approximation_rank', 1)
        self.start_powerSGD_iter = self.cache.setdefault('start_powerSGD_iter', 10)
        self.use_error_feedback = self.cache.setdefault('use_error_feedback', True)
        self.warm_start = self.cache.setdefault('warm_start', True)
        self.seed = self.cache.get('seed')
        self.error_dict = {}
        self.p_memory_dict = {}
        self.q_memory_dict = {}
        self.params_clone = {}
        self.rank1_tensors = {}
        self.high_rank_tensors = {}

        self.iter = 0

    def step(self) -> dict:
        if self.iter < self.start_powerSGD_iter:
            self.iter += 1
            return super(PowerSGDLearner, self).step()

        out = {}

        file_Qs = self.state['baseDirectory'] + _os.sep + self.input['powerSGD_Q_file_AGG']
        received_Qs = _tu.load_arrays(file_Qs)
        for k, new_q in zip(self.p_memory_dict, received_Qs):
            self.q_memory_dict[k] = new_q

        for param_key, M in self.high_rank_tensors.items():
            param = torch.matmul(self.p_memory_dict[param_key], self.q_memory_dict[param_key].t())
            if self.use_error_feedback:
                self.error_dict[param_key] = self.params_clone[param_key] - param

        # Todo

        first_optim = list(self.trainer.optimizer.keys())[0]
        self.trainer.optimizer[first_optim].step()

        return out

    def _prepare_parameters(self):
        it, out = self.backward()

        first_model = list(self.trainer.nn.keys())[0]
        for key, param in self.trainer.nn[first_model].named_parameters():
            if param.ndimension() <= 1:
                self.rank1_tensors[key] = param.detach().cpu().numpy()
            else:
                self.high_rank_tensors[key] = param.detach()

        dtype = torch.float16 if self.dtype == 'float16' else torch.float32
        for param_key, M in self.high_rank_tensors.items():
            device = M.device
            if self.use_error_feedback:
                self.params_clone[param_key] = torch.clone(M).detach
                if param_key in self.error_dict:
                    M._add(self.error_dict[param_key])
                else:
                    self.error_dict[param_key] = torch.zeros(M, device=device, dtype=dtype)

            need_randomize_qs = not self.warm_start or param_key not in self.p_memory_dict
            if need_randomize_qs:
                _orthogonalize(self.q_memory_dict[param_key])
            else:
                n, m = M.shape
                matrix_approximation_rank = min(n, m, self.matrix_approximation_rank)
                self.p_memory_dict[param_key] = torch.randn(
                    (n, matrix_approximation_rank),
                    device=device,
                    dtype=dtype,
                )
                _orthogonalize(self.q_memory_dict[param_key])

            self.p_memory_dict[param_key] = torch.matmul(M, self.q_memory_dict[param_key])

        return it, out

    def to_reduce(self):
        if self.iter < self.start_powerSGD_iter:
            it, out = super(PowerSGDLearner, self).to_reduce()
            out['start_power_iter'] = False
            return it, out

        it, out = {}, {}
        if self.input.get('powerSGD_phase', 'phase_P_sync') == 'phase_P_sync':
            """We orthogonalize P in the remote"""
            it, out = self._prepare_parameters()
            Ps = [p.detach().cpu().numpy() for p in self.p_memory_dict.values()]
            out['powerSGD_P_file'] = f"powerSGD_P_{_conf.grads_file}"
            _tu.save_arrays(
                self.state['transferDirectory'] + _sep + out['powerSGD_P_file'],
                _np.array(Ps, dtype=object)
            )

        elif self.input.get('powerSGD_phase') == 'phase_Q_sync':
            out['rank1_grads_file'] = _conf.grads_file
            _tu.save_arrays(
                self.state['transferDirectory'] + _sep + out['rank1_grads_file'],
                _np.array(list(self.rank1_tensors.values()), dtype=object)
            )

            """Recev all Ps"""
            file_Ps = self.state['baseDirectory'] + _os.sep + self.input['powerSGD_P_file_AGG']
            received_Ps = _tu.load_arrays(file_Ps)
            for k, new_p in zip(self.p_memory_dict, received_Ps):
                self.p_memory_dict[k] = new_p

            """Send all Qs"""
            for param_key, M in self.high_rank_tensors.items():
                self.q_memory_dict[param_key] = torch.matmul(M.t(), self.p_memory_dict[param_key])
            Qs = [p.detach().cpu().numpy() for p in self.q_memory_dict.values()]
            out['powerSGD_Q_file'] = f"powerSGD_Q_{_conf.grads_file}"
            _tu.save_arrays(
                self.state['transferDirectory'] + _sep + out['powerSGD_Q_file'],
                _np.array(Qs, dtype=object)
            )

        out['start_power_iter'] = True
        out['reduce'] = True
        return it, out


def _load(state, site, site_vars):
    grads_file = state['baseDirectory'] + _os.sep + site + _os.sep + site_vars['compressed_grads_file']
    return _tu.load_arrays(grads_file)


class PowerSGDReducer(_COINNReducer):

    def _average(self, recvd_file_key):
        pass

    def reduce(self):
        """ Average each sites gradients and pass it to all sites. """
        site = list(self.input.values())[0]
        if not site['start_power_iter']:
            return super(PowerSGDReducer, self).reduce()

        out = {}
        if site.get('powerSGD_P_file'):
            """Average and orthogonalize Ps"""
            out['powerSGD_phase'] = 'phase_Q_sync'

        elif site.get('powerSGD_P_file'):
            """Average Qs and rank1_grads_file"""
            out['powerSGD_phase'] = 'phase_P_sync'
            out['update'] = True

        return out
