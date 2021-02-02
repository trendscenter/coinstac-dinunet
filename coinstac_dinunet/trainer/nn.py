"""
@author: Aashis Khanal
@email: sraashis@gmail.com
@ref: https://github.com/sraashis/easytorch
"""

import json as _json
from collections import OrderedDict as _ODict
from os import sep as _sep

import torch as _torch

import coinstac_dinunet.config as _conf
import coinstac_dinunet.data as _data
import coinstac_dinunet.metrics as _base_metrics
import coinstac_dinunet.utils.tensorutils as _tu
from coinstac_dinunet.config.status import *
import coinstac_dinunet.utils as _utils
import coinstac_dinunet.vision.plotter as _plot
import math as _math


class NNTrainer:
    def __init__(self, cache: dict = None, input: dict = None, state: dict = None, **kw):
        self.cache = cache
        self.input = _utils.FrozenDict(input)
        self.state = _utils.FrozenDict(state)
        self.nn = _ODict()
        self.device = _ODict()
        self.optimizer = _ODict()

    def _init_nn_model(self):
        r"""
        User cam override and initialize required models in self.nn dict.
        """
        raise NotImplementedError('Must be implemented in child class.')

    def _init_nn_weights(self, **kw):
        r"""
        By default, will initialize network with Kaimming initialization.
        If path to pretrained weights are given, it will be used instead.
        """
        if self.cache.get('pretrained_path') is not None:
            self.load_checkpoint(self.cache['pretrained_path'])
        elif self.cache['mode'] == Mode.TRAIN:
            _torch.manual_seed(self.cache['seed'])
            for mk in self.nn:
                _tu.initialize_weights(self.nn[mk])

    def _init_optimizer(self):
        r"""
        Initialize required optimizers here. Default is Adam,
        """
        first_model = list(self.nn.keys())[0]
        self.optimizer['adam'] = _torch.optim.Adam(self.nn[first_model].parameters(),
                                                   lr=self.cache['learning_rate'])

    def init_nn(self, init_weights=False):
        self._init_nn_model()
        self._init_optimizer()
        if init_weights:
            self._init_nn_weights(init_weights=init_weights)
        self._set_gpus()

    def _set_gpus(self):
        self.device['gpu'] = _torch.device("cpu")
        if len(self.cache.get('gpus', [])) > 0:
            if _conf.gpu_available:
                self.device['gpu'] = _torch.device(f"cuda:{self.cache['gpus'][0]}")
            else:
                raise Exception(f'*** GPU not detected in {self.state["clientId"]}. ***')
        for model_key in self.nn:
            self.nn[model_key] = self.nn[model_key].to(self.device['gpu'])

    def load_checkpoint(self, file_path):
        try:
            chk = _torch.load(file_path)
        except:
            chk = _torch.load(file_path, map_location='cpu')

        if chk.get('source', 'Unknown').lower() == 'coinstac':
            for m in chk['models']:
                try:
                    self.nn[m].module.load_state_dict(chk['models'][m])
                except:
                    self.nn[m].load_state_dict(chk['models'][m])

            for m in chk['optimizers']:
                try:
                    self.optimizer[m].module.load_state_dict(chk['optimizers'][m])
                except:
                    self.optimizer[m].load_state_dict(chk['optimizers'][m])
        else:
            mkey = list(self.nn.keys())[0]
            try:
                self.nn[mkey].module.load_state_dict(chk)
            except:
                self.nn[mkey].load_state_dict(chk)

    def save_checkpoint(self, file_path, src='coinstac'):
        checkpoint = {'source': src}
        for k in self.nn:
            checkpoint['models'] = {}
            try:
                checkpoint['models'][k] = self.nn[k].module.state_dict()
            except:
                checkpoint['models'][k] = self.nn[k].state_dict()
        for k in self.optimizer:
            checkpoint['optimizers'] = {}
            try:
                checkpoint['optimizers'][k] = self.optimizer[k].module.state_dict()
            except:
                checkpoint['optimizers'][k] = self.optimizer[k].state_dict()
        _torch.save(checkpoint, file_path)

    def evaluation(self, dataset_list=None, save_pred=False):
        for k in self.nn:
            self.nn[k].eval()

        eval_avg = self.new_averages()
        eval_metrics = self.new_metrics()
        _cache = {**self.cache}
        eval_loaders = []
        for dataset in dataset_list:
            _cache['batch_size'] = _tu.get_safe_batch_size(_cache['batch_size'], len(dataset))
            eval_loaders.append(_data.COINNDataLoader.new(dataset=dataset, **_cache))

        with _torch.no_grad():
            for loader in eval_loaders:
                its = []
                metrics = self.new_metrics()
                avg = self.new_averages()
                for i, batch in enumerate(loader):
                    it = self.iteration(batch)
                    if not it.get('metrics'):
                        it['metrics'] = _base_metrics.COINNMetrics()
                    metrics.accumulate(it['metrics'])
                    avg.accumulate(it['averages'])
                    if save_pred:
                        its.append(it)
                eval_metrics.accumulate(metrics)
                eval_avg.accumulate(avg)
                if save_pred:
                    self.save_predictions(loader.dataset, its)
        return eval_avg, eval_metrics

    def train_local(self, dataset_cls, **kw):
        cache = {}
        out = {}
        self._set_monitor_metric()
        self._set_log_headers()
        cache['log_header'] = self.cache['log_header']
        cache['train_scores'] = []
        cache['validation_scores'] = []
        metric_direction = self.cache['monitor_metric'][1]
        self.cache['best_local_epoch'] = 0
        self.cache.update(best_local_score=0.0 if metric_direction == 'maximize' else _conf.data_load_lim)

        _dset_cache = {**self.cache}
        dataset = self._get_train_dataset(dataset_cls)
        _dset_cache['batch_size'] = _tu.get_safe_batch_size(_dset_cache['batch_size'], len(dataset))
        loader = _data.COINNDataLoader.new(dataset=dataset, **_dset_cache)
        epochs = self.cache.get('pretrain_epochs', self.cache['epochs'])
        for ep in range(epochs):
            for k in self.nn:
                self.nn[k].train()

            ep_averages, ep_metrics = self.new_averages(), self.new_metrics()
            _metrics, _avg = self.new_metrics(), self.new_averages()
            for i, batch in enumerate(loader):
                it = self.train_iteration_local(batch)
                if not it.get('metrics'): it['metrics'] = _base_metrics.COINNMetrics()
                ep_averages.accumulate(it['averages'])
                ep_metrics.accumulate(it['metrics'])

                _avg.accumulate(it['averages'])
                _metrics.accumulate(it['metrics'])
                if self.cache.get('verbose') and i % int(_math.log(i + 1) + 1) == 0:
                    print(f"Ep:{ep}/{epochs},Itr:{i}/{len(loader)},{_avg.get()},{_metrics.get()}")
                    _metrics.reset()
                    _avg.reset()
                self._on_iteration_end(i, ep, it)

            cache['train_scores'].append([*ep_averages.get(), *ep_metrics.get()])
            val_avg, val_metrics = self.evaluation(self._get_validation_dataset(dataset_cls), save_pred=False)
            cache['validation_scores'].append([*val_avg.get(), *val_metrics.get()])
            out.update(**self._save_if_better(ep, val_metrics))
            self._on_epoch_end(ep, ep_averages, ep_metrics, val_avg, val_metrics)
            self._plot_progress(cache)
            if self._stop_early(epoch=ep, epoch_averages=ep_averages, epoch_metrics=ep_metrics,
                                validation_averages=val_avg, validation_metric=val_metrics):
                break

        cache['best_local_epoch'] = self.cache['best_local_epoch']
        cache['best_local_score'] = self.cache['best_local_score']

        _utils.save_scores(cache, self.cache['log_dir'], file_keys=['train_scores', 'validation_scores'])
        _utils.save_cache(cache, self.cache['log_dir'])
        return out

    def iteration(self, batch):
        r"""
        Left for user to implement one mini-bath iteration:
        Example:{
                    inputs = batch['input'].to(self.device['gpu']).float()
                    labels = batch['label'].to(self.device['gpu']).long()
                    out = self.nn['model'](inputs)
                    loss = F.cross_entropy(out, labels)
                    out = F.softmax(out, 1)
                    _, pred = torch.max(out, 1)
                    sc = self.new_metrics()
                    sc.add(pred, labels)
                    avg = self.new_averages()
                    avg.add(loss.item(), len(inputs))
                    return {'loss': loss, 'averages': avg, 'output': out, 'metrics': sc, 'predictions': pred}
                }
        Note: loss, averages, and metrics are required, whereas others are optional
            -we will have to do backward on loss
            -we need to keep track of loss
            -we need to keep track of metrics
        """
        return {'metrics': _base_metrics.COINNMetrics(), 'averages': _base_metrics.COINNAverages(num_averages=1)}

    def save_predictions(self, dataset, its):
        pass

    def _reduce_iteration(self, its):
        reduced = {}.fromkeys(its[0].keys(), None)
        for k in reduced:
            if isinstance(its[0][k], _base_metrics.COINNAverages):
                reduced[k] = self.new_averages()
                [reduced[k].accumulate(ik[k]) for ik in its]

            elif isinstance(its[0][k], _base_metrics.COINNMetrics):
                reduced[k] = self.new_metrics()
                [reduced[k].accumulate(ik[k]) for ik in its]

            elif isinstance(its[0][k], _torch.Tensor) and not its[0][k].requires_grad and its[0][k].is_leaf:
                reduced[k] = _torch.cat([ik[k] for ik in its])

            else:
                reduced[k] = [ik[k] for ik in its]
        return reduced

    def _set_monitor_metric(self):
        self.cache['monitor_metric'] = 'f1', 'maximize'

    def _load_dataset(self, dataset_cls, split_key):
        dataset = dataset_cls(mode='train', limit=self.cache.get('load_limit', _conf.data_load_lim))
        file = self.cache['split_dir'] + _sep + self.cache['split_file']
        with open(file) as split_file:
            split = _json.loads(split_file.read())
            dataset.add(files=split[split_key], cache=self.cache, state=self.state)
        return dataset

    def _get_train_dataset(self, dataset_cls):
        if self.cache.get('data_indices') is None:
            return self._load_dataset(dataset_cls, split_key='train')
        dataset = dataset_cls(mode=Mode.PRE_TRAIN, limit=self.cache.get('load_limit', _conf.data_load_lim))
        dataset.indices = self.cache['data_indices']
        dataset.add(files=[], cache=self.cache, state=self.state)
        return dataset

    def _get_validation_dataset(self, dataset_cls):
        r"""
        Load the validation data from current fold/split.
        """
        return [self._load_dataset(dataset_cls, split_key='validation')]

    def _get_test_dataset(self, dataset_cls):
        r"""
        Load the test data from current fold/split.
        """
        test_dataset_list = []
        file = self.cache['split_dir'] + _sep + self.cache['split_file']
        with open(file) as split_file:
            split = _json.loads(split_file.read())
            if self.cache.get('load_sparse'):
                for f in split.get('test', []):
                    test_dataset = dataset_cls(mode='eval', limit=self.cache.get('load_limit', _conf.data_load_lim))
                    test_dataset.add(files=[f], cache=self.cache, state=self.state)
                    test_dataset_list.append(test_dataset)
            else:
                test_dataset = dataset_cls(mode='eval', limit=self.cache.get('load_limit', _conf.data_load_lim))
                test_dataset.add(files=split['test'], cache=self.cache, state=self.state)
                test_dataset_list.append(test_dataset)
        return test_dataset_list

    def _save_if_better(self, epoch, metrics):
        return {}

    def train_iteration_local(self, batch):
        first_optim = list(self.optimizer.keys())[0]
        self.optimizer[first_optim].zero_grad()
        its = []
        for _ in range(self.cache.get('local_iterations', 1)):
            it = self.iteration(batch)
            it['loss'].backward()
            its.append(it)
        self.optimizer[first_optim].step()
        return self._reduce_iteration(its)

    def new_metrics(self):
        return _base_metrics.Prf1a()

    def new_averages(self):
        return _base_metrics.COINNAverages(num_averages=1)

    def _on_epoch_end(self, ep, ep_averages, ep_metrics, val_averages, val_metrics):
        r"""
        Any logic to run after an epoch ends.
        """
        return {}

    def _on_iteration_end(self, i, ep, it):
        r"""
        Any logic to run after an iteration ends.
        """
        return {}

    def _plot_progress(self, cache, **kw):
        _plot.plot_progress(cache, self.cache['log_dir'], plot_keys=['train_scores', 'validation_scores'])

    def _stop_early(self, **kw):
        r"""
        Stop the training based on some criteria.
         For example: the implementation below will stop training if the validation
         scores does not improve within a 'patience' number of epochs.
        """
        return kw.get('epoch') - self.cache['best_local_epoch'] >= self.cache['args'].get('patience', 'epochs')

    def _set_log_headers(self):
        self.cache['log_header'] = 'Loss,Precision,Recall,F1,Accuracy'
