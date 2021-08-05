from os import sep as _sep
from typing import Tuple as _Tuple

import torch as _torch

from coinstac_dinunet import config as _conf
from coinstac_dinunet.utils import tensorutils as _tu


class COINNLearner:
    def __init__(self, trainer=None, **kw):
        self.cache = trainer.cache
        self.input = trainer.input
        self.state = trainer.state
        self.trainer = trainer
        self.global_modes = self.input.get('global_modes', {})

    def step(self) -> dict:
        out = {}
        """
        If condition checks if it is first learning step, where there is no averaged_gradient[
        available from the remote
        """
        grads = _tu.load_grads(self.state['baseDirectory'] + _sep + self.input['avg_grads_file'])

        first_model = list(self.trainer.nn.keys())[0]
        for i, param in enumerate(self.trainer.nn[first_model].parameters()):
            param.grad = _torch.tensor(grads[i], dtype=_torch.float32).to(self.trainer.device['gpu'])

        first_optim = list(self.trainer.optimizer.keys())[0]
        self.trainer.optimizer[first_optim].step()
        return out

    def backward(self) -> _Tuple[dict, dict]:
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
        return out, self.trainer.reduce_iteration(its)

    def to_reduce(self) -> _Tuple[dict, dict]:
        out, it = self.backward()
        first_model = list(self.trainer.nn.keys())[0]
        out['grads_file'] = _conf.grads_file
        grads = _tu.extract_grads(self.trainer.nn[first_model])
        _tu.save_grads(self.state['transferDirectory'] + _sep + out['grads_file'], grads)
        out['reduce'] = True
        return it, out
