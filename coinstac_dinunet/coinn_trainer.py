"""
@author: Aashis Khanal
@email: sraashis@gmail.com
@ref: https://github.com/sraashis/easytorch
"""

import json as _json
import random as _rd
from os import sep as _sep

import torch as _torch

import coinstac_dinunet.config as _conf
import coinstac_dinunet.utils.tensorutils as _tu
from coinstac_dinunet.data import COINNDataLoader as _COINNDLoader
from collections import OrderedDict as _ODict
import coinstac_dinunet.metrics as _base_metrics
import coinstac_dinunet.data as _data
from coinstac_dinunet.config.status import *


class COINNTrainer:
    def __init__(self, **kw):
        self.cache = kw['cache']
        self.input = kw['input']
        self.state = kw['state']
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
        elif self.cache('mode') == MODE_TRAIN:
            _torch.manual_seed(self.cache('seed'))
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

    def load_checkpoint_from_key(self, key='checkpoint'):
        self.load_checkpoint(self.cache['log_dir'] + _sep + self.cache[key])

    def load_checkpoint(self, full_path):
        r"""
        Load checkpoint from the given path:
            If it is an easytorch checkpoint, try loading all the models.
            If it is not, assume it's weights to a single model and laod to first model.
        """
        try:
            chk = _torch.load(full_path)
        except:
            chk = _torch.load(full_path, map_location='cpu')

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

    def backward(self, it):
        out = {}
        it['loss'].backward()
        out['grads_file'] = _conf.grads_file
        first_model = list(self.nn.keys())[0]
        _tu.save_grads(self.state['transferDirectory'] + _sep + out['grads_file'],
                       grads=_tu.extract_grads(self.nn[first_model]))
        return out

    def step(self):
        grads = _tu.load_grads(self.state['baseDirectory'] + _sep + self.input['avg_grads_file'])

        first_model = list(self.nn.keys())[0]
        for i, param in enumerate(self.nn[first_model].parameters()):
            param.grad = _torch.tensor(grads[i], dtype=_torch.float32).to(self.device['gpu'])

        first_optim = list(self.optimizer.keys())[0]
        self.optimizer[first_optim].step()

    def save_checkpoint(self, file_name, src='coinstac'):
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
        _torch.save(checkpoint, self.cache['log_dir'] + _sep + file_name)

    def evaluation(self, dataset_list=None, save_pred=False):
        r"""
        Evaluation phase that handles validation/test phase
        split-key: the key to list of files used in this particular evaluation.
        The program will create k-splits(json files) as per specified in --nf -num_of_folds
         argument with keys 'train', ''validation', and 'test'.
        """
        for k in self.nn:
            self.nn[k].eval()

        eval_avg = self.new_averages()
        eval_metrics = self.new_metrics()
        eval_loaders = [_data.COINNDataLoader.new(dataset=d, mode='eval', **self.cache) for d in dataset_list]
        with _torch.no_grad():
            for loader in eval_loaders:
                its = []
                metrics = self.new_metrics()
                avg = self.new_averages()
                for i, batch in enumerate(loader):
                    it = self.iteration(batch)
                    if not it.get('metrics'):
                        it['metrics'] = _base_metrics.ETMetrics()
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
        return {'metrics': _base_metrics.ETMetrics(), 'averages': _base_metrics.ETAverages(num_averages=1)}

    def save_predictions(self, dataset, its):
        r"""
        If one needs to save complex predictions result like predicted segmentations.
         -Especially with U-Net architectures, we split images and train.
        Once the argument --sp/-sparse-load is set to True,
        the argument 'its' will receive all the patches of single image at a time.
        From there, we can recreate the whole image.
        """
        pass

    def train(self, dataset_cls):
        out = {}

        for k in self.nn:
            self.nn[k].train()
        for k in self.optimizer:
            self.optimizer[k].zero_grad()

        itr_avgs, itr_metrics = self.new_averages(), self.new_metrics()

        its = []
        for _ in range(self.cache.get('local_iterations', 1)):
            it = self.iteration(self.next_batch(dataset_cls, mode='train'))
            out.update(**self.backward(it))
            out.update(**self.next_iter())
            itr_avgs.accumulate(it['averages'])
            itr_metrics.accumulate(it['metrics'])
            its.append([it])
        self.cache['train_log'].append([vars(itr_avgs), vars(itr_metrics)])

        if self.input.get('avg_grads_file'):
            self.step()
            self.save_checkpoint(file_name=self.cache['current_nn_state'])

        out.update(**self._on_iteration_end(0, self.cache['epoch'], its))
        return out

    def validation(self, dataset_cls):
        out = {}
        avg, scores = self.evaluation(self._get_validation_dataset(dataset_cls), save_pred=False)
        out['validation_log'] = [vars(avg), vars(scores)]
        out.update(**self.next_epoch())
        out.update(**self._on_epoch_end(self.cache['epoch'], None, None, avg, scores))
        return out

    def test(self, dataset_cls):
        out = {}
        self.load_checkpoint_from_key(key='best_nn_state')
        avg, scores = self.evaluation(self._get_test_dataset(dataset_cls), save_pred=True)
        out['test_log'] = [vars(avg), vars(scores)]
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
        split_file = open(self.cache['split_dir'] + _sep + self.cache['split_file']).read()
        with open(split_file) as split_file:
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
            val_dataset = dataset_cls(mode='eval', limit=self.cache.get('load_limit', float('inf')))
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
                    test_dataset = dataset_cls(mode='eval', limit=self.cache.get('load_limit', float('inf')))
                    test_dataset.add(files=[f], cache=self.cache, state=self.state)
                    test_dataset_list.append(test_dataset)
            else:
                test_dataset = dataset_cls(mode='eval', limit=self.cache.get('load_limit', float('inf')))
                test_dataset.add(files=split['test'], cache=self.cache, state=self.state)
                test_dataset_list.append(test_dataset)
        return test_dataset_list

    def new_metrics(self):
        r"""
        User can override to supply desired implementation of easytorch.metrics.ETMetrics().
            Example: easytorch.metrics.Pr11a() will work with precision, recall, F1, Accuracy, IOU scores.
        """
        return _base_metrics.Prf1a()

    def new_averages(self):
        r""""
        Should supply an implementation of easytorch.metrics.ETAverages() that can keep track of multiple averages.
            Example: multiple loss, or any other values.
        """
        return _base_metrics.ETAverages(num_averages=1)

    def next_iter(self):
        out = {}
        self.cache['cursor'] += self.cache['batch_size']
        if self.cache['cursor'] >= self.cache['data_len']:
            out['mode'] = MODE_VALIDATION_WAITING
            self.cache['cursor'] = 0
            _rd.shuffle(self.cache['data_indices'])
        return out

    def next_batch(self, dataset_cls, mode='train'):
        dataset = dataset_cls(mode='eval', limit=self.cache.get('load_limit', float('inf')))
        dataset.indices = self.cache['data_indices'][self.cache['cursor']:]
        loader = _COINNDLoader.new(mode=mode, **self.cache)
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
            out['mode'] = MODE_TEST
        else:
            self.cache['cursor'] = 0
            out['mode'] = MODE_TRAIN_WAITING
            _rd.shuffle(self.cache['data_indices'])

        out['train_log'] = self.cache['train_log']
        self.cache['train_log'] = []
        return out

    def _on_epoch_end(self, ep, ep_loss, ep_metrics, val_loss, val_metrics):
        r"""
        Any logic to run after an epoch ends.
        """
        return {}

    def _on_iteration_end(self, i, ep, it):
        r"""
        Any logic to run after an iteration ends.
        """
        return {}