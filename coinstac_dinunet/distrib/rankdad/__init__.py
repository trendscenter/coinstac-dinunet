import numpy as _np
import os as _os

from coinstac_dinunet.distrib.learner import COINNLearner as _COINNLearner
from coinstac_dinunet.distrib.reducer import COINNReducer as _COINNReducer
from coinstac_dinunet.utils import tensorutils as _tu
import coinstac_dinunet.config as _config
from .spi import DADParallel as DADParallel, power_iteration_BC
import torch as _torch


class DADLearner(_COINNLearner):
    def __init__(self, **kw):
        super().__init__(**kw)
        for fk in self.trainer.nn:
            self.trainer.nn[fk] = DADParallel(
                self.trainer.nn[fk],
                cache=self.cache,
                input=self.input,
                state=self.state,
                device=self.trainer.device['gpu'],
                dtype=self.dtype
            )

    def step(self):
        out = {}
        first_model = list(self.trainer.nn.keys())[0]
        self.trainer.nn[first_model].synced_param_update()
        first_optim = list(self.trainer.optimizer.keys())[0]
        self.trainer.optimizer[first_optim].step()
        return out

    def forward(self):
        out = {}
        first_model = list(self.trainer.nn.keys())[0]
        first_optim = list(self.trainer.optimizer.keys())[0]

        self.trainer.nn[first_model].train()
        self.trainer.optimizer[first_optim].zero_grad()

        its = []
        for _ in range(self.cache['local_iterations']):
            batch, nxt_iter_out = self.trainer.data_handle.next_iter()
            it = self.trainer.iteration(batch)
            it['loss'].backward()
            its.append(it)
            out.update(**nxt_iter_out)
            """Cannot use grad accumulation with DAD at the moment"""
            break
        return self.trainer.reduce_iteration(its), out

    def to_reduce(self):
        it, out = {}, {}
        fk = list(self.trainer.nn.keys())[0]
        self.trainer.nn[fk].train()
        it, fw_out = self.forward()
        out.update(**fw_out)
        out.update(**self.trainer.nn[fk].dad_backward())
        out['reduce'] = True
        return it, out


class DADReducer(_COINNReducer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.rank = self.cache.setdefault('dad_reduction_rank', 10)
        self.num_pow_iters = self.cache.setdefault('dad_num_pow_iters', 5)
        self.dad_tol = self.cache.setdefault('dad_tol', 1e-3)

    def reduce(self):
        out = {'reduced_dad_data': 'reduced_dad_data.npy'}

        site_data = self._load("dad_data")
        gpu_data = []
        for d in list(zip(*site_data)):
            grad, act = list(zip(*d))
            grad = _torch.cat([_torch.from_numpy(g).to(self.device, non_blocking=True) for g in grad], 1).squeeze(-1)
            act = _torch.cat([_torch.from_numpy(a).to(self.device, non_blocking=True) for a in act], 1)
            if _config.CUDA_AVAILABLE:
                grad, act = power_iteration_BC(
                    grad, act,
                    self.rank, self.num_pow_iters, self.dad_tol
                )
            gpu_data.append([grad, act])

        reduced_data = []
        for grad, act in gpu_data:
            reduced_data.append([
                grad.cpu().numpy().astype(self.dtype),
                act.cpu().numpy().astype(self.dtype)
            ])

        _tu.save_arrays(
            self.state['transferDirectory'] + _os.sep + out['reduced_dad_data'],
            _np.array(reduced_data, dtype=object)
        )
        out['update'] = True
        return out
