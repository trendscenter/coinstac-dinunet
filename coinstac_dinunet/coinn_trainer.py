"""
@author: Aashis Khanal
@email: sraashis@gmail.com
@ref: https://github.com/sraashis/easytorch
"""

import json as _json
import random as _rd
from collections import OrderedDict as _ODict
from os import sep as _sep

import torch as _torch

import coinstac_dinunet.config as _conf
import coinstac_dinunet.data as _data
import coinstac_dinunet.metrics as _base_metrics
import coinstac_dinunet.utils.tensorutils as _tu
from coinstac_dinunet.config.status import *
from coinstac_dinunet.data import COINNDataLoader as _COINNDLoader
from coinstac_dinunet.utils import FrozenDict as _FrozenDict


class COINNTrainer:
    def __init__(self, cache: dict = None, input: dict = None, state: dict = None, **kw):
        self.cache = cache
        self.input = _FrozenDict(input)
        self.state = _FrozenDict(state)
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

    def step(self):
        grads = _tu.load_grads(self.state['baseDirectory'] + _sep + self.input['avg_grads_file'])

        first_model = list(self.nn.keys())[0]
        for i, param in enumerate(self.nn[first_model].parameters()):
            param.grad = _torch.tensor(grads[i], dtype=_torch.float32).to(self.device['gpu'])

        first_optim = list(self.optimizer.keys())[0]
        self.optimizer[first_optim].step()

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
        eval_loaders = [_data.COINNDataLoader.new(dataset=d, **self.cache) for d in dataset_list]
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

    def _save_if_better(self, metrics):
        r"""
        Save the current model as best if it has better validation scores.
        """
        out = {}
        monitor_metric, direction = self.cache['monitor_metric']
        sc = getattr(metrics, monitor_metric)
        if callable(sc):
            sc = sc()

        if (direction == 'maximize' and sc >= self.cache['best_local_score']) or (
                direction == 'minimize' and sc <= self.cache['best_local_score']):
            out['pretrained_weights'] = _conf.pretrained_weights_file
            self.save_checkpoint(file_path=self.state['transferDirectory'] + _sep + out['pretrained_weights'])
        return out

    def train_local(self, dataset_cls, num_sites=1, **kw):
        out = {}
        self._set_monitor_metric()
        out['pretrain_scores'] = []
        metric_direction = self.cache['monitor_metric'][1]
        self.cache.update(best_local_score=0 if metric_direction == 'maximize' else 1e11)
        with open(self.cache['log_dir'] + _sep + 'pretrained.csv', 'w') as writer:
            first_model = list(self.nn.keys())[0]
            first_optim = list(self.optimizer.keys())[0]

            self.nn[first_model].train()
            self.optimizer[first_optim].zero_grad()

            dataset = dataset_cls(mode=Mode.PRE_TRAIN, limit=self.cache.get('load_limit', _conf.data_load_lim))

            dataset.indices = self.cache['data_indices']
            dataset.add(files=[], cache=self.cache, state=self.state)
            loader = _COINNDLoader.new(dataset=dataset, **self.cache)
            for ep in range(self.cache['pretrain_epochs']):
                ep_averages = self.new_averages()
                ep_metrics = self.new_metrics()
                for i, batch in enumerate(loader):
                    its = []
                    for _ in range(self.cache['local_iterations']):
                        it = self.iteration(batch)
                        it['loss'].backward()
                        its.append(it)
                    self.optimizer[first_optim].step()
                    it = self._reduce_iteration(its)
                    self._on_iteration_end(i, ep, it)
                    ep_averages.accumulate(it['averages'])
                    ep_metrics.accumulate(it['metrics'])

                writer.write(f'{[*ep_averages.get()]}, {[*ep_metrics.get()]}\n')
                writer.flush()
                out['pretrain_scores'].append([*ep_averages.get(), *ep_metrics.get()])
                val_avg, val_metrics = self.evaluation(self._get_validation_dataset(dataset_cls), save_pred=False)
                out.update(**self._save_if_better(val_metrics))
                self._on_epoch_end(ep, ep_averages, ep_metrics, val_avg, val_metrics)
        return out

    def train_distributed(self, dataset_cls):
        out = {}

        first_model = list(self.nn.keys())[0]
        first_optim = list(self.optimizer.keys())[0]

        self.nn[first_model].train()
        self.optimizer[first_optim].zero_grad()

        if self.input.get('avg_grads_file'):
            self.step()
            self.save_checkpoint(file_path=self.cache['log_dir'] + _sep + self.cache['current_nn_state'])

        its = []
        for _ in range(self.cache['local_iterations']):
            it = self.iteration(self.next_batch(dataset_cls, mode='train'))
            it['loss'].backward()
            its.append(it)
            out.update(**self.next_iter())
        it = self._reduce_iteration(its)

        out['grads_file'] = _conf.grads_file
        grads = _tu.extract_grads(self.nn[first_model])
        _tu.save_grads(self.state['transferDirectory'] + _sep + out['grads_file'], grads)
        self.cache['train_scores'].append([vars(it['averages']), vars(it['metrics'])])
        out.update(**self._on_iteration_end(0, self.cache['epoch'], it))
        return out

    def validation(self, dataset_cls):
        out = {}
        avg, scores = self.evaluation(self._get_validation_dataset(dataset_cls), save_pred=False)
        out['validation_scores'] = [vars(avg), vars(scores)]
        out.update(**self.next_epoch())
        out.update(**self._on_epoch_end(self.cache['epoch'], None, None, avg, scores))
        return out

    def test(self, dataset_cls):
        out = {}
        self.load_checkpoint(self.cache['log_dir'] + _sep + self.cache['best_nn_state'])
        avg, scores = self.evaluation(self._get_test_dataset(dataset_cls), save_pred=True)
        out['test_scores'] = [vars(avg), vars(scores)]
        return out

    def load_data_indices(self, dataset_cls, split_key='train'):
        """
        Parse and load dataset and save to cache:
        so that in next global iteration we dont have to do that again.
        The data IO depends on use case-For a instance, if your data can fit in RAM, you can load
         and save the entire dataset in cache. But, in general,
         it is better to save indices in cache, and load only a mini-batch at a time
         (logic in __nextitem__) of the data loader.
        """
        dataset = dataset_cls(mode='eval', limit=self.cache.get('load_limit', float('inf')))
        file = self.cache['split_dir'] + _sep + self.cache['split_file']
        with open(file) as split_file:
            split = _json.loads(split_file.read())
            dataset.add(files=split[split_key], cache=self.cache, state=self.state)

        self.cache['data_indices'] = dataset.indices
        if len(dataset) % self.cache['batch_size'] >= self.cache.get('min_batch_size', 4):
            self.cache['data_len'] = len(dataset)
        else:
            self.cache['data_len'] = (len(dataset) // self.cache['batch_size']) * self.cache['batch_size']

    def _get_validation_dataset(self, dataset_cls):
        r"""
        Load the validation data from current fold/split.
        """
        val_dataset_list = []
        file = self.cache['split_dir'] + _sep + self.cache['split_file']
        with open(file) as split_file:
            split = _json.loads(split_file.read())
            val_dataset = dataset_cls(mode='eval', limit=self.cache.get('load_limit', _conf.data_load_lim))
            val_dataset.add(files=split['validation'], cache=self.cache, state=self.state)
            val_dataset_list.append(val_dataset)
        return val_dataset_list

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

    def new_metrics(self):
        return _base_metrics.Prf1a()

    def new_averages(self):
        return _base_metrics.COINNAverages(num_averages=1)

    def next_iter(self):
        out = {}
        self.cache['cursor'] += self.cache['batch_size']
        if self.cache['cursor'] >= self.cache['data_len']:
            out['mode'] = Mode.VALIDATION_WAITING
            self.cache['cursor'] = 0
            _rd.shuffle(self.cache['data_indices'])
        return out

    def next_batch(self, dataset_cls, mode='train'):
        dataset = dataset_cls(mode=mode, limit=self.cache.get('load_limit', _conf.data_load_lim))
        dataset.indices = self.cache['data_indices'][self.cache['cursor']:]
        dataset.add(files=[], cache=self.cache, state=self.state)
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

        out['train_scores'] = self.cache['train_scores']
        self.cache['train_scores'] = []
        return out

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
