"""
@author: Aashis Khanal
@email: sraashis@gmail.com
@ref: https://github.com/sraashis/easytorch
"""

import numpy as _np
import torch as _torch

import coinstac_dinunet.config as _conf


def safe_concat(large, small):
    r"""
    Safely concat two slightly unequal tensors.
    """
    diff = _np.array(large.shape) - _np.array(small.shape)
    diffa = _np.floor(diff / 2).astype(int)
    diffb = _np.ceil(diff / 2).astype(int)

    t = None
    if len(large.shape) == 4:
        t = large[:, :, diffa[2]:large.shape[2] - diffb[2], diffa[3]:large.shape[3] - diffb[3]]
    elif len(large.shape) == 5:
        t = large[:, :, diffa[2]:large.shape[2] - diffb[2], diffa[3]:large.shape[3] - diffb[3],
            diffa[4]:large.shape[2] - diffb[4]]

    return _torch.cat([t, small], 1)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, _torch.nn.Conv2d) or isinstance(module, _torch.nn.Linear):
                _torch.nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, _torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def caste_tensor(a):
    if _conf.grad_file_format == '.npy':
        return a.numpy().astype(f'float{_conf.grad_precision_bit}')
    elif _conf.grad_file_format == '.tar':
        if _conf.grad_precision_bit == 16:
            return a.half()
        elif _conf.grad_precision_bit == 32:
            return a.float()
        elif _conf.grad_precision_bit == 64:
            return a.double()
    return a


def extract_grads(model):
    return [caste_tensor(p.grad.detach().cpu()) for p in model.parameters()]


def save_grads(file_name, grads):
    if _conf.grads_numpy:
        _np.save(file_name, _np.asarray(grads))
    elif _conf.grads_torch:
        _torch.save(grads, file_name)


def load_grads(file_name):
    grads = None
    if _conf.grads_numpy:
        grads = _np.load(file_name, allow_pickle=True)
    if _conf.grads_torch:
        grads = _torch.load(file_name)
    return grads