import glob as _glob
import os as _os
import shutil as _shu
from typing import Tuple as _Tuple

import numpy as _np
import torch as _torch
from coinstac_dinunet.config.keys import Mode

from coinstac_dinunet.distrib.learner import COINNLearner
from coinstac_dinunet.distrib.reducer import COINNReducer


def check(logic, k, v, kw):
    phases = []
    for site_vars in kw.values():
        phases.append(site_vars.get(k) == v)
    return logic(phases)


def hook_wrapper(site, hook_type, layer, save_to='', debug=False):
    if debug:
        print(f"**** {site}, {hook_type}, {layer} ****")

    name = _os.path.join(save_to, f"Type-{hook_type}_Layer-{layer}")

    def save(m, in_grad, out_grad):
        if hook_type.lower() == 'forward':
            for i, b in enumerate(in_grad):
                if b is not None:
                    _np.save(name + f"_IO-in_Index-{i}.npy", b.clone().detach().numpy())
                break
        if hook_type.lower() == 'backward':
            for i, c in enumerate(out_grad):
                if c is not None:
                    _np.save(name + f"_IO-out_Index-{i}.npy", c.clone().detach().numpy())
                break

    return save


class DADLearner(COINNLearner):
    DATA_PATH = '_dad_layers_data'
    GRADS_PATH = '_dad_weights_update'

    def __init__(self, **kw):
        super().__init__(**kw)
        self.log_dir = self.state['outputDirectory'] + _os.sep + DADLearner.DATA_PATH
        self.grads_dir = self.state['outputDirectory'] + _os.sep + DADLearner.GRADS_PATH

        if self.global_modes[self.state['clientId']] == Mode.TRAIN:
            for model_key in self.trainer.nn.keys():
                for layer, ch in list(self.trainer.nn[model_key].named_children()):
                    ch.register_forward_hook(hook_wrapper(self.state['clientId'], 'forward', layer, self.log_dir))
                    ch.register_backward_hook(hook_wrapper(self.state['clientId'], 'backward', layer, self.log_dir))

    def step(self) -> dict:
        if self.input.get('dad_reduced_layer'):
            self.dad_backward()

        out = {}
        out['save_state'] = False
        if self.input.get('dad_step'):
            grads = []
            for layer, _ in self.trainer.nn[list(self.trainer.nn.keys())[0]].named_parameters():
                grads.append(_torch.tensor(_np.load(self.grads_dir + _os.sep + f"{layer}_grad.npy").T))

            for i, param in enumerate(self.trainer.nn[list(self.trainer.nn.keys())[0]].parameters()):
                param.grad = grads[i].to(self.trainer.device['gpu'])

            first_optim = list(self.trainer.optimizer.keys())[0]
            self.trainer.optimizer[first_optim].step()
            out['save_state'] = True
        return out

    def forward(self, dataset_cls) -> _Tuple[dict, dict]:
        out = {}

        first_model = list(self.trainer.nn.keys())[0]
        first_optim = list(self.trainer.optimizer.keys())[0]

        self.trainer.nn[first_model].train()
        self.trainer.optimizer[first_optim].zero_grad()

        _os.makedirs(self.log_dir, exist_ok=True)
        _os.makedirs(self.grads_dir, exist_ok=True)
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
        """Update each layer's grads after getting aggregate from remote"""

        act = _torch.FloatTensor(_np.load(self.state['baseDirectory'] + _os.sep + self.input['dad_layer_activation']),
                                 device=self.trainer.device['gpu'])
        grad = _torch.FloatTensor(_np.load(self.state['baseDirectory'] + _os.sep + self.input['dad_layer_grads']),
                                  device=self.trainer.device['gpu'])
        _np.save(self.grads_dir + _os.sep + self.input['dad_reduced_layer'] + ".weight_grad.npy",
                 (act.T.mm(grad)).cpu().numpy())

    def to_reduce(self, dataset_cls) -> _Tuple[dict, dict]:
        out, it = {}, {}
        if len(self.cache.get('dad_layers', [])) == 0:

            """Clear for next global iteration"""
            if _os.path.exists(self.log_dir):
                _shu.rmtree(self.log_dir)

            if _os.path.exists(self.grads_dir):
                _shu.rmtree(self.grads_dir)

            out, it = self.forward(dataset_cls)
            fk = list(self.trainer.nn.keys())[0]
            self.cache['dad_layers'] = [k for k, v in self.trainer.nn[fk].named_children()]
            out['last_layer'] = True

        if len(self.cache['dad_layers']) > 0:
            self.cache['dad_layer'] = self.cache['dad_layers'].pop()
            layer_files = _glob.glob(self.log_dir + _os.sep + f"*_Layer-{self.cache['dad_layer']}_*")
            for file in layer_files:
                _shu.move(file, self.state['transferDirectory'])
            out['dad_layer'] = self.cache['dad_layer']

        out['to_step'] = len(self.cache['dad_layers']) == 0
        out['reduce'] = True
        return out, it


class DADReducer(COINNReducer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def reduce_VGG(self):
        out = {}
        h, grad_prev = [], []
        for site, site_vars in self.input.items():
            fw_file = self.state[
                          'baseDirectory'] + _os.sep + site + _os.sep + f"Type-forward_Layer-{site_vars['dad_layer']}_IO-in_Index-0.npy"
            bk_file = self.state[
                          'baseDirectory'] + _os.sep + site + _os.sep + f"Type-backward_Layer-{site_vars['dad_layer']}_IO-out_Index-0.npy"
            h.append(_np.load(fw_file))
            grad_prev.append(_np.load(bk_file))

        out["dad_layer_activation"] = site_vars['dad_layer'] + "_activation.npy"
        out["dad_layer_grads"] = site_vars['dad_layer'] + "_grads.npy"

        _np.save(self.state['transferDirectory'] + _os.sep + out['dad_layer_activation'], _np.concatenate(h))
        _np.save(self.state['transferDirectory'] + _os.sep + out['dad_layer_grads'], _np.concatenate(grad_prev))

        out['dad_reduced_layer'] = site_vars['dad_layer']
        out['update'] = True
        out['dad_step'] = check(all, 'to_step', True, self.input)
        return out

    def reduce(self):
        return self.reduce_VGG()
