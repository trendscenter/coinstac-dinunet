"""
@author: Aashis Khanal
@email: sraashis@gmail.com
@ref: https://github.com/sraashis/easytorch
"""

import random as _rd
from collections import OrderedDict as _ODict
from os import sep as _sep

import torch as _torch

import coinstac_dinunet.config as _conf
import coinstac_dinunet.utils as _utils
import coinstac_dinunet.utils.tensorutils as _tu
from coinstac_dinunet.config.status import *
from coinstac_dinunet.data import COINNDataLoader as _COINNDLoader
from .nn import NNTrainer as _NNTrainer


class COINNTrainer(_NNTrainer):
    def __init__(self, cache: dict = None, input: dict = None, state: dict = None, **kw):
        super().__init__(cache, input, state, **kw)
        self.cache = cache
        self.input = _utils.FrozenDict(input)
        self.state = _utils.FrozenDict(state)
        self.nn = _ODict()
        self.device = _ODict()
        self.optimizer = _ODict()

    def step(self):
        grads = _tu.load_grads(self.state['baseDirectory'] + _sep + self.input['avg_grads_file'])

        first_model = list(self.nn.keys())[0]
        for i, param in enumerate(self.nn[first_model].parameters()):
            param.grad = _torch.tensor(grads[i], dtype=_torch.float32).to(self.device['gpu'])

        first_optim = list(self.optimizer.keys())[0]
        self.optimizer[first_optim].step()

    def _save_if_better(self, epoch, metrics):
        r"""
        Save the current model as best if it has better validation scores.
        """
        out = {}
        monitor_metric, direction = self.cache['monitor_metric']
        sc = getattr(metrics, monitor_metric)
        if callable(sc):
            sc = sc()

        if (direction == 'maximize' and sc > self.cache['best_local_score']) or (
                direction == 'minimize' and sc < self.cache['best_local_score']):
            out['weights_file'] = _conf.weights_file
            self.cache['best_local_epoch'] = epoch
            self.cache['best_local_score'] = sc
            self.save_checkpoint(file_path=self.state['transferDirectory'] + _sep + out['weights_file'])
        return out

    def train_distributed(self, dataset_cls):
        out = {}

        first_model = list(self.nn.keys())[0]
        first_optim = list(self.optimizer.keys())[0]

        self.nn[first_model].train()
        self.optimizer[first_optim].zero_grad()

        its = []
        for _ in range(self.cache['local_iterations']):
            it = self.iteration(self.next_batch(dataset_cls))
            it['loss'].backward()
            its.append(it)
            out.update(**self.next_iter())
        it = self._reduce_iteration(its)

        out['grads_file'] = _conf.grads_file
        grads = _tu.extract_grads(self.nn[first_model])
        _tu.save_grads(self.state['transferDirectory'] + _sep + out['grads_file'], grads)
        self.cache[Key.TRAIN_SERIALIZABLE].append([vars(it['averages']), vars(it['metrics'])])
        out.update(**self._on_iteration_end(0, self.cache['epoch'], it))
        return out

    def validation_distributed(self, dataset_cls):
        out = {}
        avg, metrics = self.evaluation(mode='validation_dist', save_pred=False,
                                       dataset_list=[self._get_validation_dataset(dataset_cls)])
        out[Key.VALIDATION_SERIALIZABLE] = [vars(avg), vars(metrics)]
        out.update(**self.next_epoch())
        out.update(**self._on_epoch_end(self.cache['epoch'], None, None, avg, metrics))
        out['epoch'] = self.cache['epoch']
        return out

    def test_distributed(self, dataset_cls):
        out = {}
        self.load_checkpoint(self.cache['log_dir'] + _sep + self.cache['best_nn_state'])
        avg, metrics = self.evaluation(mode='dist_test', save_pred=True,
                                       dataset_list=[self._get_test_dataset(dataset_cls)])
        out[Key.TEST_SERIALIZABLE] = [vars(avg), vars(metrics)]
        out['epoch'] = self.cache['epoch']
        return out

    def cache_data_indices(self, dataset_cls, split_key='train'):
        dataset = self._load_dataset(dataset_cls, split_key)
        self.cache['data_indices'] = dataset.indices
        if len(dataset) % self.cache['batch_size'] >= self.cache.get('min_batch_size', 4):
            self.cache['data_len'] = len(dataset)
        else:
            self.cache['data_len'] = (len(dataset) // self.cache['batch_size']) * self.cache['batch_size']

    def next_iter(self):
        out = {}
        self.cache['cursor'] += self.cache['batch_size']
        if self.cache['cursor'] >= self.cache['data_len']:
            out['mode'] = Mode.VALIDATION_WAITING
            self.cache['cursor'] = 0
            _rd.shuffle(self.cache['data_indices'])
        return out

    def next_batch(self, dataset_cls):
        dataset = self._get_train_dataset(dataset_cls)
        dataset.indices = dataset.indices[self.cache['cursor']:]
        loader = _COINNDLoader.new(dataset=dataset, **self.cache)
        return next(loader.__iter__())

    def next_epoch(self):

        """
        Transition to next epoch after validation.
        It will set 'train_waiting' status if we need more training
        Else it will set 'test' status
        """
        out = {}
        self.cache['epoch'] += 1
        if self.cache['epoch'] - self.cache.get('best_epoch', self.cache['epoch']) \
                >= self.cache['patience'] or self.cache['epoch'] >= self.cache['epochs']:
            out['mode'] = Mode.TEST
        else:
            self.cache['cursor'] = 0
            out['mode'] = Mode.TRAIN_WAITING
            _rd.shuffle(self.cache['data_indices'])

        out[Key.TRAIN_SERIALIZABLE] = self.cache[Key.TRAIN_SERIALIZABLE]
        self.cache[Key.TRAIN_SERIALIZABLE] = []
        return out
