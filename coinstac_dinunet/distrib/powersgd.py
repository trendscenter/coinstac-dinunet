import os as _os

import torch as _torch

from coinstac_dinunet import config as _conf
from coinstac_dinunet.utils import tensorutils as _tu
import numpy as _np
from coinstac_dinunet.distrib.learner import COINNLearner as _COINNLearner
from coinstac_dinunet.distrib.reducer import COINNReducer as _COINNReducer
from functools import partial as _partial

_sep = _os.sep


class PowerSGDLearner(_COINNLearner):

    def step(self) -> dict:
        out = {}
        """
        If condition checks if it is first learning step, where there is no averaged_gradient[
        available from the remote
        """
        grads = _tu.load_arrays(self.state['baseDirectory'] + _sep + self.input['agg_grads_file'])

        first_model = list(self.trainer.nn.keys())[0]
        for i, param in enumerate(self.trainer.nn[first_model].parameters()):
            param.grad = _torch.tensor(grads[i], dtype=_torch.float32).to(self.trainer.device['gpu'])

        first_optim = list(self.trainer.optimizer.keys())[0]
        self.trainer.optimizer[first_optim].step()
        return out

    def backward(self):
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
        return self.trainer.reduce_iteration(its), out

    def to_reduce(self):
        it, out = self.backward()
        first_model = list(self.trainer.nn.keys())[0]
        out['compressed_grads_file'] = _conf.grads_file
        grads = _tu.extract_grads(self.trainer.nn[first_model], dtype=self.dtype)

        compressed_grads = []  # PowerSGD compression todo

        _tu.save_arrays(
            self.state['transferDirectory'] + _sep + out['compressed_grads_file'],
            _np.array(compressed_grads, dtype=object)
        )
        out['reduce'] = True
        return it, out


def _load(state, site, site_vars):
    grads_file = state['baseDirectory'] + _os.sep + site + _os.sep + site_vars['compressed_grads_file']
    return _tu.load_arrays(grads_file)


class COINNReducer(_COINNReducer):

    def reduce(self):
        """ Average each sites gradients and pass it to all sites. """
        out = {'avg_grads_file': _conf.avg_grads_file}

        grads = list(
            self.pool.starmap(
                _partial(_load, self.state), self.input.items()
            )
        )

        # Todo
        return_grads = []
        _tu.save_arrays(
            self.state['transferDirectory'] + _os.sep + out['agg_grads_file'],
            _np.array(return_grads, dtype=object)
        )

        out['update'] = True
        return out
