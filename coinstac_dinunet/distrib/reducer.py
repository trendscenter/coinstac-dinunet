import os as _os
from functools import partial as _partial

import numpy as _np

import coinstac_dinunet.config as _conf
import coinstac_dinunet.utils.tensorutils as _tu
import torch as _torch


def _load(state, site, site_vars):
    grads_file = state['baseDirectory'] + _os.sep + site + _os.sep + site_vars['grads_file']
    return _tu.load_arrays(grads_file)


class COINNReducer:
    def __init__(self, trainer, mp_pool, **kw):
        self.cache = trainer.cache
        self.input = trainer.input
        self.state = trainer.state
        self.trainer = trainer
        self.pool = mp_pool
        self.dtype = f"float{self.cache['precision_bits']}"

    def reduce(self):
        """ Average each sites gradients and pass it to all sites. """
        out = {'avg_grads_file': _conf.avg_grads_file}

        grads = list(
            self.pool.starmap(
                _partial(_load, self.state), self.input.items()
            )
        )

        avg_grads = []
        for data in list(zip(*grads)):
            data = _torch.from_numpy(_np.array(data)).to(_conf.DEVICE).mean(0)
            avg_grads.append(data.cpu().numpy().astype(self.dtype))

        _tu.save_arrays(
            self.state['transferDirectory'] + _os.sep + out['avg_grads_file'],
            _np.array(avg_grads, dtype=object)
        )

        out['update'] = True
        return out
