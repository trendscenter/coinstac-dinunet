import coinstac_dinunet.config as _conf
import os as _os
import coinstac_dinunet.utils.tensorutils as _tu
import numpy as _np


class COINNReducer:
    def __init__(self, trainer, **kw):
        self.cache = trainer.cache
        self.input = trainer.input
        self.state = trainer.state
        self.trainer = trainer

    def reduce(self):
        """ Average each sites gradients and pass it to all sites. """
        out = {'avg_grads_file': _conf.avg_grads_file}
        grads = []
        for site, site_vars in self.input.items():
            grads_file = self.state['baseDirectory'] + _os.sep + site + _os.sep + site_vars['grads_file']
            grads.append(_tu.load_arrays(grads_file))

        avg_grads = []
        for layer_grad in zip(*grads):
            avg_grads.append(_np.array(layer_grad).mean(0))
        _tu.save_arrays(self.state['transferDirectory'] + _os.sep + out['avg_grads_file'], avg_grads)
        out['update'] = True
        return out
