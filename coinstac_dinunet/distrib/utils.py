import glob as _glob
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
        if hook_type.lower() == 'forward':
            for i, b in enumerate(in_grad):
                if b is not None:
                    _np.save(name + f"-IO:in-Index:{i}.npy", b.clone().detach().numpy())
                break
        if hook_type.lower() == 'backward':
            for i, c in enumerate(out_grad):
                if c is not None:
                    _np.save(name + f"-IO:out-index:{i}.npy", c.clone().detach().numpy())
                break

    return hook_save


def check(logic, k, v, kw):
    phases = []
    for site_vars in kw.values():
        phases.append(site_vars.get(k) == v)
    return logic(phases)


class DADLearner(COINNLearner):
    DATA_PATH = '_dad_data'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.log_dir = self.cache['outputDirectory'] + _os.sep + DADLearner.DATA_PATH
        for model_key in self.trainer.nn.keys():
            for layer, ch in list(self.trainer.nn[model_key].children())[::-1]:
                ch.register_forward_hook(hook_wrapper(self.state['clientId'], 'forward', layer, self.log_dir))
                ch.register_backward_hook(hook_wrapper(self.state['clientId'], 'backward', layer, self.log_dir))

    def step(self) -> dict:
        if self.input.get('dad_layer'):
            """First layer since step is called before dad_backward"""
            self.dad_backward()
        out = {}
        out['save_state'] = False
        if self.input.get('dad_step'):
            # Todo
            out['save_state'] = True
        return out

    def forward(self, dataset_cls) -> _Tuple[dict, dict]:
        out = {}

        first_model = list(self.trainer.nn.keys())[0]
        first_optim = list(self.trainer.optimizer.keys())[0]

        self.trainer.nn[first_model].train()
        self.trainer.optimizer[first_optim].zero_grad()

        _os.makedirs(self.log_dir, exist_ok=True)
        its = []
        """Cannot use grad accumulation with DAD at the moment"""
        # for _ in range(self.cache['local_iterations']):
        for _ in range(1):
            it = self.trainer.iteration(self.trainer.next_batch(dataset_cls))
            it['loss'].backward()
            its.append(it)
            out.update(**self.trainer.next_iter())
        return out, self.trainer.reduce_iteration(its)

    def dad_backward(self):
        #Todo
        """Update each layer's grads after getting aggregate from remote"""
        pass

    def to_reduce(self, dataset_cls) -> _Tuple[dict, dict]:
        out, it = {}, {}
        if len(self.cache.get('dad_layers', [])) == 0:

            """Clear for next global iteration"""
            if _os.path.exists(self.log_dir):
                _shu.rmtree(self.log_dir)

            out, it = self.forward(dataset_cls)
            fk = list(self.trainer.nn.keys())[0]
            self.cache['dad_layers'] = [k for k, v in self.trainer.nn[fk].children()]
            out['last_layer'] = True

        if len(self.cache['dad_layers']) > 0:
            self.cache['dad_layer'] = self.cache['dad_layers'].pop()
            layer_files = _glob.glob(self.log_dir + _os.sep + f"*-Layer:{self.cache['dad_iter']}-*")
            transfer_path = self.state['transferDirectory'] + _os.sep + DADLearner.DATA_PATH
            _os.makedirs(transfer_path)
            for file in layer_files:
                _shu.move(file, transfer_path)
            out['dad_layer'] = self.cache['dad_layer']

        out['to_step'] = len(self.cache['dad_layers']) == 0
        out['reduce'] = True
        return out, it


class DADReducer(COINNReducer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def reduce(self):
        """ Average each sites gradients and pass it to all sites. """
        out = {}
        for site, site_vars in self.input.items():
            re = f"*-Layer:{self.cache['dad_iter']}-*"
            layer_files = _glob.glob(
                self.state['baseDirectory'] + _os.sep + site + _os.sep + DADLearner.DATA_PATH + _os.sep + re
            )

            if site_vars.get('last_layer'):
                """Vertical concat"""
                pass
            else:
                """Vertical concat"""
                pass

        out['dad_layer'] = site_vars.get('dad_layer')
        out['update'] = True
        out['dad_step'] = check(all, 'to_step', True, self.input)
        return out
