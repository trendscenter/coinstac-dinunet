import random as _rd
from os import sep as _sep

import torch as _torch

from coinstac_dinunet import config as _conf
from coinstac_dinunet.config.state import *
from coinstac_dinunet.utils import tensorutils as _tu


class CoinnLearner:
    def __init__(self, trainer=None, **kw):
        self.cache = trainer.cache
        self.input = trainer.input
        self.state = trainer.state
        self.trainer = trainer

    def step(self) -> dict:
        out = {'save_state': False}
        """If condition checks if it is first learning step, where there is no averaged_gradient available from the remote"""
        if self.input.get('avg_grads_file'):
            grads = _tu.load_grads(self.state['baseDirectory'] + _sep + self.input['avg_grads_file'])

            first_model = list(self.trainer.nn.keys())[0]
            for i, param in enumerate(self.trainer.nn[first_model].parameters()):
                param.grad = _torch.tensor(grads[i], dtype=_torch.float32).to(self.trainer.device['gpu'])

            first_optim = list(self.trainer.optimizer.keys())[0]
            self.trainer.optimizer[first_optim].step()
            out['save_state'] = True

        return out

    def backward(self, dataset_cls) -> dict:
        out = {}

        first_model = list(self.trainer.nn.keys())[0]
        first_optim = list(self.trainer.optimizer.keys())[0]

        self.trainer.nn[first_model].train()
        self.trainer.optimizer[first_optim].zero_grad()

        its = []
        for _ in range(self.cache['local_iterations']):
            it = self.trainer.iteration(self.trainer.next_batch(dataset_cls))
            it['loss'].backward()
            its.append(it)
            out.update(**self.trainer.next_iter())
        it = self.trainer.reduce_iteration(its)

        out['grads_file'] = _conf.grads_file
        grads = _tu.extract_grads(self.trainer.nn[first_model])
        _tu.save_grads(self.state['transferDirectory'] + _sep + out['grads_file'], grads)
        self.cache[Key.TRAIN_SERIALIZABLE].append([vars(it['averages']), vars(it['metrics'])])
        out.update(**self.trainer.on_iteration_end(0, 0, it))
        return out


class PowerSGDLearner(CoinnLearner):
    def __init__(self, trainer=None, **kw):
        super().__init__(trainer=trainer, **kw)
