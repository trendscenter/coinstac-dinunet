import multiprocessing as mp
import os as _os

import numpy as _np

import coinstac_dinunet.config as _conf
import coinstac_dinunet.utils.tensorutils as _tu
from functools import partial as _partial


def _load(state, site, site_vars):
    grads_file = state['baseDirectory'] + _os.sep + site + _os.sep + site_vars['grads_file']
    return _tu.load_arrays(grads_file)


def _mean(*data):
    return _np.array(data).mean(0)


class COINNReducer:
    def __init__(self, trainer, **kw):
        self.cache = trainer.cache
        self.input = trainer.input
        self.state = trainer.state
        self.trainer = trainer
        self.num_workers = min(4, len(self.input)//2)

    def reduce(self):
        """ Average each sites gradients and pass it to all sites. """
        out = {'avg_grads_file': _conf.avg_grads_file}

        with mp.Pool(processes=self.num_workers) as pool:
            grads = list(pool.starmap(_partial(_load, self.state), self.input.items()))

        with mp.Pool(processes=self.num_workers) as pool:
            avg_grads = list(pool.starmap(_mean, list(zip(*grads))))
            _tu.save_arrays(self.state['transferDirectory'] + _os.sep + out['avg_grads_file'], avg_grads)
            out['update'] = True

        return out
