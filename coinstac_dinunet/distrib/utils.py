import os as _os
import shutil as _shu
from typing import Tuple as _Tuple

import numpy as _np

from coinstac_dinunet import COINNLearner, COINNReducer


def hook_wrapper(site, hook_type, layer, save_to='', debug=False):
    if debug:
        print(f"**** {site}, {hook_type}, {layer} ****")

    name = _os.path.join(save_to, f"Site:{site}-Type:{hook_type}-Layer:{layer}")

    def hook_save(a, in_grad, out_grad):
        for i, b in enumerate(in_grad):
            if b is not None:
                _np.save(name + f"-IO:in-Index:{i}.npy", b.clone().detach().numpy())
        if hook_type.lower() == 'backward':
            for i, c in enumerate(out_grad):
                if c is not None:
                    _np.save(name + f"-IO:out-index:{i}.npy", c.clone().detach().numpy())

    return hook_save


class DADLearner(COINNLearner):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.base_dir_fw = self.state['outputDirectory'] + _os.sep + '_dad_temp'
        self.base_dir_bk = self.state['outputDirectory'] + _os.sep + '_dad_temp'
        for model_key in self.trainer.nn.keys():
            for layer, ch in list(self.trainer.nn[model_key].children())[::-1]:
                ch.register_forward_hook(hook_wrapper(self.state['clientId'], 'forward', layer, self.base_dir_fw))
                ch.register_backward_hook(hook_wrapper(self.state['clientId'], 'backward', layer, self.base_dir_bk))

    def step(self) -> dict:
        out = {}
        # Todo
        out['save_state'] = True
        return out

    def fw_backward(self, dataset_cls) -> _Tuple[dict, dict]:
        out = {}

        first_model = list(self.trainer.nn.keys())[0]
        first_optim = list(self.trainer.optimizer.keys())[0]

        self.trainer.nn[first_model].train()
        self.trainer.optimizer[first_optim].zero_grad()

        _os.makedirs(self.base_dir_fw, exist_ok=True)
        _os.makedirs(self.base_dir_bk, exist_ok=True)

        its = []
        """Cannot use grad accumulation with DAD at the moment"""
        # for _ in range(self.cache['local_iterations']):
        for _ in range(1):
            it = self.trainer.iteration(self.trainer.next_batch(dataset_cls))
            it['loss'].backward()
            its.append(it)
            out.update(**self.trainer.next_iter())
        return out, self.trainer.reduce_iteration(its)

    def to_reduce(self, dataset_cls) -> _Tuple[dict, dict]:
        out, it = {}, {}
        if len(self.cache.get('dad_iterations', [])) == 0:

            """Clear for next global iteration"""
            if _os.path.exists(self.base_dir_fw):
                _shu.rmtree(self.base_dir_fw)
            if _os.path.exists(self.base_dir_bk):
                _shu.rmtree(self.base_dir_bk)

            out, it = self.fw_backward(dataset_cls)
            fk = list(self.trainer.nn.keys())[0]
            self.cache['dad_iterations'] = [k for k, v in self.trainer.nn[fk].children()]

        if len(self.cache['dad_iterations']) > 0:
            self.cache['dad_iter'] = self.cache['dad_iterations'].pop()
            """Todo for sending layer's data"""
            out['dad_iter'] = self.cache['dad_iter']
            out['to_step'] = len(self.cache['dad_iterations']) == 0

        out['reduce'] = True
        return out, it


class DADReducer(COINNReducer):
    pass
