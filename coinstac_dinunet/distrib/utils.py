import numpy as _np
import os as _os
from functools import partial as _partial

from coinstac_dinunet.distrib.learner import COINNLearner as _COINNLearner
from coinstac_dinunet.distrib.reducer import COINNReducer as _COINNReducer
from coinstac_dinunet.utils import tensorutils as _tu
from .dad import DADParallel as _DADParallel


def check(logic, k, v, kw):
    phases = []
    for site_vars in kw.values():
        phases.append(site_vars.get(k) == v)
    return logic(phases)


def _load(state, site, site_vars):
    grads_file = state['baseDirectory'] + _os.sep + site + _os.sep + site_vars['dad_data']
    return _tu.load_arrays(grads_file)


def _cat(*data):
    act, grad = list(zip(*data))
    return [_np.concatenate(act), _np.concatenate(grad)]


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
                reduction_rank=self.cache.get('dad_reduction_rank', 10),
                num_pow_iters=self.cache.get('num_pow_iters', 5)
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
        if not self.cache.get('reduction_chunk_size'):
            self.cache['reduction_chunk_size'] = _math.ceil(len(self.input) / self.pool._processes)

    def reduce(self):
        out = {'reduced_dad_data': 'reduced_dad_data.npy'}

        site_data = list(self.pool.starmap(_partial(_load, self.state), self.input.items(),
                                           chunksize=self.cache['reduction_chunk_size']))

        reduced_data = list(
            self.pool.starmap(_cat, list(zip(*site_data)), chunksize=self.cache['reduction_chunk_size']))
        _tu.save_arrays(self.state['transferDirectory'] + _os.sep + out['reduced_dad_data'], reduced_data)
        out['update'] = True
        return out
