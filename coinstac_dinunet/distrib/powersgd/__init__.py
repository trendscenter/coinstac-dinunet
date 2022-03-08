import os as _os

import numpy as _np
import torch as _torch
from coinstac_dinunet import config as _conf
from coinstac_dinunet.utils import tensorutils as _tu

from ..learner import COINNLearner as _COINNLearner
from ..reducer import COINNReducer as _COINNReducer
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import _orthogonalize
from functools import partial as _partial

_sep = _os.sep


class PowerSGDState:
    def __init__(self):
        self.error_dict = {}
        self.p_memory_dict = {}
        self.q_memory_dict = {}
        self.params_clone = {}
        self.rank1_tensors = {}
        self.high_rank_tensors = {}
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
        file_Qs = self.powerSGD_state['baseDirectory'] + _os.sep + self.input['powerSGD_Q_file_AGG']
        received_Qs = _tu.load_arrays(file_Qs)
        for k, new_q in zip(self.powerSGD_state.p_memory_dict, received_Qs):
            self.powerSGD_state.q_memory_dict[k] = _torch.from_numpy(new_q).to(self.device)

        file_rank_1 = self.powerSGD_state['baseDirectory'] + _os.sep + self.input['rank_1_grads_file_AGG']
        rank_1_params = [_torch.from_numpy(t).to(self.device).float() for t in _tu.load_arrays(file_rank_1)]

        high_rank_params = []
        for param_key, M in self.powerSGD_state.high_rank_tensors.items():
            recon_param = _torch.matmul(self.powerSGD_state.p_memory_dict[param_key],
                                        self.powerSGD_state.q_memory_dict[param_key].t()).float()
            high_rank_params.append(recon_param)
            if self.use_error_feedback:
                self.powerSGD_state.error_dict[param_key] = self.powerSGD_state.params_clone[param_key] - recon_param

        first_model = list(self.trainer.nn.keys())[0]

        rank_1_params = rank_1_params[::-1]
        high_rank_params = high_rank_params[::-1]
        for param_key, param in self.trainer.nn[first_model].named_parameters():
            if param_key in self.powerSGD_state.rank1_tensors:
                param.grad = rank_1_params.pop()

            elif param_key in high_rank_params:
                param.grad = high_rank_params.pop()

        first_optim = list(self.trainer.optimizer.keys())[0]
        self.trainer.optimizer[first_optim].step()
        return out

    def _prepare_parameters(self):
        it, out = self.backward()

        first_model = list(self.trainer.nn.keys())[0]
        for key, param in self.trainer.nn[first_model].named_parameters():
            if param.ndimension() <= 1:
                self.powerSGD_state.rank1_tensors[key] = param.grad.detach().cpu().numpy().astype(self.dtype)
            else:
                self.powerSGD_state.high_rank_tensors[key] = param.grad.detach()

        for param_key, M in self.powerSGD_state.high_rank_tensors.items():
            if self.use_error_feedback:
                self.powerSGD_state.params_clone[param_key] = _torch.clone(M).detach()
                if param_key in self.powerSGD_state.error_dict:
                    M._add(self.powerSGD_state.error_dict[param_key])
                else:
                    self.powerSGD_state.error_dict[param_key] = _torch.zeros(M.shape, device=self.device, dtype=M.dtype)

            need_randomize_qs = not self.warm_start or param_key not in self.powerSGD_state.p_memory_dict
            if need_randomize_qs:
                n, m = M.shape
                self.powerSGD_state.q_memory_dict[param_key] = _torch.randn(
                    (m, self.matrix_approximation_rank),
                    device=self.device,
                    dtype=M.dtype,
                )
                _orthogonalize(self.powerSGD_state.q_memory_dict[param_key])
            else:
                _orthogonalize(self.powerSGD_state.q_memory_dict[param_key])

            self.powerSGD_state.p_memory_dict[param_key] = _torch.matmul(M,
                                                                         self.powerSGD_state.q_memory_dict[param_key])

        return it, out

    def to_reduce(self):
        if self.powerSGD_state.iter < self.start_powerSGD_iter:
            it, out = super(PowerSGDLearner, self).to_reduce()
            out['start_power_iter'] = False
            return it, out

        it, out = {}, {}
        if self.input.get('powerSGD_phase', 'phase_P_sync') == 'phase_P_sync':
            it, out = self._prepare_parameters()
            Ps = [p.detach().cpu().numpy().astype(self.dtype) for p in self.powerSGD_state.p_memory_dict.values()]
            out['powerSGD_P_file'] = f"powerSGD_P_{_conf.grads_file}"
            _tu.save_arrays(
                self.state['transferDirectory'] + _sep + out['powerSGD_P_file'],
                _np.array(Ps, dtype=object)
            )

        elif self.input.get('powerSGD_phase') == 'phase_Q_sync':
            out['rank1_grads_file'] = f"rank1_{_conf.grads_file}"
            _tu.save_arrays(
                self.state['transferDirectory'] + _sep + out['rank1_grads_file'],
                _np.array(list(self.powerSGD_state.rank1_tensors.values()), dtype=object)
            )

            """Recev all Ps"""
            file_Ps = self.state['baseDirectory'] + _os.sep + self.input['powerSGD_P_file_AGG']
            received_Ps = _tu.load_arrays(file_Ps)
            for k, new_p in zip(self.powerSGD_state.p_memory_dict, received_Ps):
                new_p = _torch.from_numpy(new_p).to(self.device)
                _orthogonalize(new_p)
                self.powerSGD_state.p_memory_dict[k] = new_p

            """Send all Qs"""
            for param_key, M in self.powerSGD_state.high_rank_tensors.items():
                self.powerSGD_state.q_memory_dict[param_key] = _torch.matmul(
                    M.t(),
                    self.powerSGD_state.p_memory_dict[param_key]
                )
            Qs = [p.detach().cpu().numpy().astype(self.dtype) for p in self.powerSGD_state.q_memory_dict.values()]
            out['powerSGD_Q_file'] = f"powerSGD_Q_{_conf.grads_file}"
            _tu.save_arrays(
                self.state['transferDirectory'] + _sep + out['powerSGD_Q_file'],
                _np.array(Qs, dtype=object)
            )

        out['start_power_iter'] = True
        out['reduce'] = True
        return it, out


def _load(state, file_key, site, site_vars):
    grads_file = state['baseDirectory'] + _os.sep + site + _os.sep + site_vars[file_key]
    return _tu.load_arrays(grads_file)


class PowerSGDReducer(_COINNReducer):

    def _average(self, file_key):
        grads = list(
            self.pool.starmap(
                _partial(_load, self.state, file_key), self.input.items()
            )
        )

        avg_grads = []
        for data in list(zip(*grads)):
            data = _torch.from_numpy(_np.array(data)).to(self.device).mean(0)
            avg_grads.append(data.cpu().numpy().astype(self.dtype))
        return avg_grads

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

        elif site.get('powerSGD_P_file'):
            """Average Qs and rank1_grads_file"""

            out['rank_1_grads_file_AGG'] = f"rank1_AGG_{_conf.avg_grads_file}"
            _tu.save_arrays(
                self.state['transferDirectory'] + _os.sep + out['powerSGD_P_file_AGG'],
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
