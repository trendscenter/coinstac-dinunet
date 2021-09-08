import os as _os

import numpy as _np

from coinstac_dinunet.distrib.learner import COINNLearner as _COINNLearner
from coinstac_dinunet.distrib.reducer import COINNReducer as _COINNReducer
from .dad import DADParallel as _DADParallel


def check(logic, k, v, kw):
    phases = []
    for site_vars in kw.values():
        phases.append(site_vars.get(k) == v)
    return logic(phases)


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
                reduction_rank=self.cache.get('dad_reduction_rank', 7),
                num_pow_iters=self.cache.get('num_pow_iters', 3)
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
        out = {}
        out['tall_grads_file'] = {}
        out['tall_activation_file'] = {}
        site_var = list(self.input.values())[0]
        for layer_name, layer in site_var['dad_layers'].items():
            h, delta = [], []
            out['tall_activation_file'][layer_name] = f"{layer_name}_activation_tall.npy"
            out['tall_grads_file'][layer_name] = f"{layer_name}_grads_tall.npy"
            for site, site_var in self.input.items():
                _layer = site_var['dad_layers'][layer_name]
                h.append(
                    _np.load(self.state['baseDirectory'] + _os.sep + site + _os.sep + _layer['activation_file'])
                )
                delta.append(
                    _np.load(self.state['baseDirectory'] + _os.sep + site + _os.sep + _layer['grads_file'])
                )
            _np.save(self.state['transferDirectory'] + _os.sep + out['tall_activation_file'][layer_name],
                     _np.concatenate(h))
            _np.save(self.state['transferDirectory'] + _os.sep + out['tall_grads_file'][layer_name],
                     _np.concatenate(delta))
        out['update'] = True
        return out
