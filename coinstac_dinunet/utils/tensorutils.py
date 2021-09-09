"""
@author: Aashis Khanal
@email: sraashis@gmail.com
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


def caste_ndarray(a):
    return a.astype(f'float{_conf.grad_precision_bit}')


def extract_grads(model):
    return [caste_ndarray(p.grad.detach().cpu().numpy()) for p in model.parameters()]


def save_arrays(file_path, arrays):
    _np.save(file_path, arrays)


def load_arrays(file_path):
    return _np.load(file_path, allow_pickle=True)
