"""
@author: Aashis Khanal
@email: sraashis@gmail.com
"""

import random as _rd
from collections import OrderedDict as _ODict
from os import sep as _sep

import coinstac_dinunet.config as _conf
import coinstac_dinunet.utils as _utils
from coinstac_dinunet.config.keys import *
from coinstac_dinunet.utils.utils import performance_improved_
from .nn import NNTrainer as _NNTrainer
from coinstac_dinunet import COINNDataLoader as _COINNDLoader


class COINNTrainer(_NNTrainer):
    def __init__(self, cache: dict = None, input: dict = None, state: dict = None, **kw):
        super().__init__(cache, input, state, **kw)
        self.cache = cache
        self.input = _utils.FrozenDict(input)
        self.state = _utils.FrozenDict(state)
        self.nn = _ODict()
        self.device = _ODict()
        self.optimizer = _ODict()

    def _save_if_better(self, epoch, val_metrics):
        r"""
        Save the current model as best if it has better validation scores in pretraining step.
        """
        out = {}
        val_score = val_metrics.extract(self.cache['monitor_metric'][0])
        if performance_improved_(epoch, val_score, self.cache):
            out['weights_file'] = _conf.weights_file
            self.save_checkpoint(file_path=self.state['transferDirectory'] + _sep + out['weights_file'])
        return out

    def validation_distributed(self, dataset_cls):
        out = {}
        avg, metrics = self.evaluation(mode='validation', save_pred=False,
                                       dataset_list=self._get_validation_dataset_list(dataset_cls))
        out[Key.VALIDATION_SERIALIZABLE] = [vars(avg), vars(metrics)]
        out[Key.TRAIN_SERIALIZABLE] = self.cache[Key.TRAIN_SERIALIZABLE]
        self.cache[Key.TRAIN_SERIALIZABLE] = []
        _rd.shuffle(self.cache.get('data_indices', []))
        self.cache['cursor'] = 0
        return out

    def test_distributed(self, dataset_cls):
        out = {}
        self.load_checkpoint(self.cache['log_dir'] + _sep + self.cache['best_nn_state'])
        avg, metrics = self.evaluation(mode='test', save_pred=True,
                                       dataset_list=self._get_test_dataset_list(dataset_cls))
        out[Key.TEST_SERIALIZABLE] = [vars(avg), vars(metrics)]
        return out

    def cache_data_indices(self, dataset_cls, split_key='train'):
        dataset = self._load_dataset(dataset_cls, split_key)
        self.cache['data_indices'] = dataset.indices
        self.cache['data_len'] = (len(dataset) // self.cache['batch_size']) * self.cache['batch_size']

    def next_batch(self, dataset_cls):
        dataset = self._get_train_dataset(dataset_cls)
        dataset.indices = dataset.indices[self.cache['cursor']:]
        loader = _COINNDLoader.new(dataset=dataset, **self.cache)
        return next(loader.__iter__())

    def next_iter(self) -> dict:
        out = {}
        self.cache['cursor'] += self.cache['batch_size']
        if self.cache['cursor'] >= self.cache['data_len']:
            out['mode'] = Mode.VALIDATION_WAITING
            _rd.shuffle(self.cache['data_indices'])
            self.cache['cursor'] = 0
        return out
