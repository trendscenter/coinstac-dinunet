import os as _os
from functools import partial as _partial

import numpy as _np

import coinstac_dinunet.config as _conf
import coinstac_dinunet.utils.tensorutils as _tu
import torch as _torch


def _multi_load(file_key, state, site, site_vars):
    grads_file = state['baseDirectory'] + _os.sep + site + _os.sep + site_vars[file_key]
    return _tu.load_arrays(grads_file)


class COINNReducer:

    def _load(self, file_key):
        return list(
            self.pool.starmap(
                _partial(_multi_load, file_key, self.state), self.input.items()
            )
        )

    def _average(self, file_key):
        sites_data = self._load(file_key)
        gpu_data = []
        for data in list(zip(*sites_data)):
            avg = _torch.from_numpy(_np.array(data)).to(self.device, non_blocking=True).mean(0)
            gpu_data.append(avg)

        return [data.cpu().numpy().astype(self.dtype) for data in gpu_data]

    def __init__(self, trainer, mp_pool, **kw):
        self.cache = trainer.cache
        self.input = trainer.input
        self.state = trainer.state
        self.trainer = trainer
        self.pool = mp_pool
        self.dtype = f"float{self.cache['precision_bits']}"
        self.device = trainer.device['gpu']

    def reduce(self):
        """ Average each sites gradients and pass it to all sites. """
        out = {
            'avg_grads_file': _conf.avg_grads_file
        }
        _tu.save_arrays(
            self.state['transferDirectory'] + _os.sep + out['avg_grads_file'],
            _np.array(self._average('grads_file'), dtype=object)
        )

        out['update'] = True
        return out
