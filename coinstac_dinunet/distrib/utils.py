import numpy as _np
import os as _os
from functools import partial as _partial

from coinstac_dinunet.distrib.learner import COINNLearner as _COINNLearner
from coinstac_dinunet.distrib.reducer import COINNReducer as _COINNReducer
from coinstac_dinunet.utils import tensorutils as _tu
import coinstac_dinunet.config as _config
from .dad import DADParallel as _DADParallel, power_iteration_BC as _SPI
import torch as _torch


def _load(state, site, site_vars):
    grads_file = state['baseDirectory'] + _os.sep + site + _os.sep + site_vars['dad_data']
    return _tu.load_arrays(grads_file)


def _cat(dtype, cache, *data):
    """
    Starmap calls by doing func(*data) itself so don't have to do data[0]
    """
    act, grad = list(zip(*data))
    act = _torch.cat([_torch.from_numpy(a).to(_config.DEVICE) for a in act])
    grad = _torch.cat([_torch.from_numpy(g).to(_config.DEVICE) for g in grad], 1).squeeze(0)
    if _config.CUDA_AVAILABLE:
        grad, act = _SPI(
            grad.T, act.T,
            rank=cache.setdefault('dad_reduction_rank', 10),
            numiterations=cache.setdefault('dad_num_pow_iters', 5),
            device=_config.DEVICE,
            tol=cache.setdefault('dad_tol', 1e-3)
        )
        act, grad = act.T, grad.T
    return [act.cpu().numpy().astype(dtype), grad.unsqueeze(0).cpu().numpy().astype(dtype)]


class DADLearner(_COINNLearner):
    def __init__(self, **kw):
        super().__init__(**kw)
        for fk in self.trainer.nn:
            self.trainer.nn[fk] = _DADParallel(
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

    def reduce(self):
        out = {'reduced_dad_data': 'reduced_dad_data.npy'}

        site_data = list(
            self.pool.starmap(
                _partial(_load, self.state), self.input.items(),
                chunksize=self._chunk_size
            )
        )

        reduced_data = list(
            self.pool.starmap(
                _partial(_cat, self.dtype, self.cache), list(zip(*site_data)),
                chunksize=self._chunk_size
            )
        )

        _tu.save_arrays(
            self.state['transferDirectory'] + _os.sep + out['reduced_dad_data'],
            _np.array(reduced_data, dtype=object)
        )
        out['update'] = True
        return out
