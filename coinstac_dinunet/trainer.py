"""
@author: Aashis Khanal
@email: sraashis@gmail.com
"""

import random as _rd
from os import sep as _sep

import coinstac_dinunet.config as _conf
import coinstac_dinunet.utils as _utils
from coinstac_dinunet.config.keys import *
from coinstac_dinunet.utils.utils import performance_improved_
from .nn import NNTrainer as _NNTrainer


class COINNTrainer(_NNTrainer):
    def __init__(self, cache: dict = None, input: dict = None, state: dict = None, **kw):
        super().__init__(cache, input, state, **kw)
        self.cache = cache
        self.input = _utils.FrozenDict(input)
        self.state = _utils.FrozenDict(state)
        self.nn = self.cache.get('nn', {})
        self.device = self.cache.get('device', {})
        self.optimizer = self.cache.get('optimizer', {})

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

    def validation_distributed(self):
        out = {}
        validation_dataset = self.data_handle.dataset['validation']
        if not isinstance(validation_dataset, list):
            validation_dataset = [validation_dataset]

        avg, metrics = self.evaluation(
            mode='validation',
            save_pred=False,
            dataset_list=validation_dataset
        )
        out[Key.VALIDATION_SERIALIZABLE] = [vars(avg), vars(metrics)]
        out[Key.TRAIN_SERIALIZABLE] = self.cache[Key.TRAIN_SERIALIZABLE]
        self.cache[Key.TRAIN_SERIALIZABLE] = []
        _rd.shuffle(self.data_handle.dataset['train'].indices)
        self.cache['cursor'] = 0
        return out

    def test_distributed(self):
        out = {}
        self.load_checkpoint(self.cache['log_dir'] + _sep + self.cache['best_nn_state'])
        test_dataset = self.data_handle.get_test_dataset()
        if not isinstance(test_dataset, list):
            test_dataset = [test_dataset]

        avg, metrics = self.evaluation(mode='test', save_pred=True,
                                       dataset_list=test_dataset)
        out[Key.TEST_SERIALIZABLE] = [vars(avg), vars(metrics)]
        return out

    def next_batch(self):
        if self.cache['cursor'] == 0:
            dataset = self.data_handle.dataset['train']
            self.cache['train_loader_iter'] = iter(
                self.data_handle.get_loader(handle_key='train', dataset=dataset, shuffle=True)
            )
        return next(self.cache['train_loader_iter'])

    def next_iter(self) -> dict:
        out = {}
        self.cache['cursor'] += self.cache['batch_size']
        if self.cache['cursor'] >= self.cache['data_len']:
            out['mode'] = Mode.VALIDATION_WAITING
            _rd.shuffle(self.data_handle.dataset['train'].indices)
            self.cache['cursor'] = 0
        return out
