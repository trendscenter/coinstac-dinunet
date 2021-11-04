import math as _math
import os as _os
from functools import partial as _partial

import numpy as _np

import coinstac_dinunet.config as _conf
import coinstac_dinunet.utils.tensorutils as _tu


def _load(state, site, site_vars):
    grads_file = state['baseDirectory'] + _os.sep + site + _os.sep + site_vars['grads_file']
    return _tu.load_arrays(grads_file)


def _mean(dtype, *data):
    """
    Starmap calls by doing func(*data) itself so dont have to do data[0]
    """
    return _np.array(data).mean(0).astype(dtype)


class COINNReducer:
    def __init__(self, trainer, mp_pool, **kw):
        self.cache = trainer.cache
        self.input = trainer.input
        self.state = trainer.state
        self.trainer = trainer
        self.pool = mp_pool
        self.dtype = f"float{self.cache['precision_bits']}"

        self._chunk_size = self.cache.setdefault(
            'reduction_chunk_size',
            _math.ceil(len(self.input) / mp_pool._processes)
        )

    def reduce(self):
        """ Average each sites gradients and pass it to all sites. """
        out = {'avg_grads_file': _conf.avg_grads_file}

        grads = list(
            self.pool.starmap(
                _partial(_load, self.state), self.input.items(),
                chunksize=self._chunk_size
            )
        )

        avg_grads = list(
            self.pool.starmap(
                _partial(_mean, self.dtype), list(zip(*grads)),
                chunksize=self._chunk_size
            )
        )

        _tu.save_arrays(
            self.state['transferDirectory'] + _os.sep + out['avg_grads_file'],
            _np.array(avg_grads, dtype=object)
        )

        out['update'] = True
        return out
