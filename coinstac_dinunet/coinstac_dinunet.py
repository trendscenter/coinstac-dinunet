"""
@author: Aashis Khanal
@email: sraashis@gmail.com
@ref: https://github.com/sraashis/easytorch
"""

import random as _rd
from os import sep as _sep

from torch.nn import functional as _F
import coinstac_dinunet.metrics as _base_metrics
import coinstac_dinunet.data as _dt
import coinstac_dinunet.utils.tensorutils as _tu
import numpy as _np
import json as _json
import torch as _torch
import coinstac_dinunet.config as _cs


class COINNIter:
    def __init__(self, **kw):
        self.cache = kw['cache']
        self.input = kw['input']
        self.state = kw['state']

    def next_iter(self):
        out = {}
        self.cache['cursor'] += self.cache['batch_size']
        if self.cache['cursor'] >= self.cache['data_len']:
            out['mode'] = 'val_waiting'
            self.cache['cursor'] = 0
            _rd.shuffle(self.cache['data_indices'])
        return out

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
            out['mode'] = 'test'
        else:
            self.cache['cursor'] = 0
            out['mode'] = 'train_waiting'
            _rd.shuffle(self.cache['data_indices'])

        out['train_log'] = self.cache['train_log']
        self.cache['train_log'] = []
        return out


class COINNTrainer(COINNIter):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.cache = kw['cache']
        self.input = kw['input']
        self.state = kw['state']
        self.nn = {}
        self.optimizer = {}
        self.device = {}

    def init_nn(self, init_weights=False):
        self._init_nn_model()
        self._init_optimizer()
        if init_weights:
            for mk in self.nn:
                _torch.manual_seed(self.cache['seed'])
                _tu.initialize_weights(self.nn[mk])
        self._set_gpus()

    def _init_nn_model(self):
        r"""
        User cam override and initialize required models in self.core dict.
        """
        raise NotImplementedError('Must be implemented in child class.')

    def _init_optimizer(self):
        r"""
        Initialize required optimizers here. Default is Adam,
        """
        first_model = list(self.nn.keys())[0]
        self.optimizer['adam'] = _torch.optim.Adam(self.nn[first_model].parameters(),
                                                   lr=self.cache['learning_rate'])

    def _set_gpus(self):
        self.device['gpu'] = _torch.device("cpu")
        if len(self.cache.get('gpus', [])) > 0:
            if _torch.cuda.is_available():
                self.device['gpu'] = _torch.device(f"cuda:{self.cache['gpus'][0]}")
            else:
                raise Exception(f'*** GPU not detected in {self.state["clientId"]}. ***')
        for model_key in self.nn:
            self.nn[model_key] = self.nn[model_key].to(self.device['gpu'])

    def new_metrics(self):
        return _base_metrics.Prf1a()

    def new_averages(self):
        return _base_metrics.ETAverages()

    def backward(self, it):
        out = {}
        it['loss'].backward()
        out['grads_file'] = _cs.grads_file

        first_model = list(self.nn.keys())[0]
        if _cs.is_format_numpy:
            grads = [p.grad.type(_torch.float16).detach().cpu().numpy() for p in self.nn[first_model].parameters()]
            _np.save(self.state['transferDirectory'] + _sep + out['grads_file'], _np.asarray(grads))
        elif _cs.is_format_torch:
            grads = [p.grad.type(_cs.float_precision) for p in self.nn[first_model].parameters()]
            _torch.save(grads, self.state['transferDirectory'] + _sep + out['grads_file'])

        self.cache['train_log'].append([vars(it['avg_loss']), vars(it['score'])])
        return out

    def step(self):
        if _cs.is_format_numpy:
            avg_grads = _np.load(self.state['baseDirectory'] + _sep + self.input['avg_grads'], allow_pickle=True)
        if _cs.is_format_torch:
            avg_grads = _torch.load(self.state['baseDirectory'] + _sep + self.input['avg_grads'])

        first_model = list(self.nn.keys())[0]
        for i, param in enumerate(self.nn[first_model].parameters()):
            param.grad = _torch.tensor(avg_grads[i], dtype=_torch.float32).to(self.device['gpu'])

        first_optim = list(self.optimizer.keys())[0]
        self.optimizer[first_optim].step()

    def save_checkpoint(self, name):
        checkpoint = {'source': "coinstac"}
        for k in self.nn:
            checkpoint['nn'] = {}
            try:
                checkpoint['nn'][k] = self.nn[k].module.state_dict()
            except:
                checkpoint['nn'][k] = self.nn[k].state_dict()

        for k in self.optimizer:
            checkpoint['optimizer'] = {}
            try:
                checkpoint['optimizer'][k] = self.optimizer[k].module.state_dict()
            except:
                checkpoint['optimizer'][k] = self.optimizer[k].state_dict()
        _torch.save(checkpoint, self.cache['log_dir'] + _sep + name)

    def _load_checkpoint(self, name):
        r"""
        Load checkpoint from the given path:
            If it is an easytorch checkpoint, try loading all the models.
            If it is not, assume it's weights to a single model and laod to first model.
        """
        full_path = _torch.load(self.cache['log_dir'] + _sep + name)
        try:
            chk = _torch.load(full_path)
        except:
            chk = _torch.load(full_path, map_location='cpu')

        if chk.get('source', 'Unknown').lower() == 'coinstac':
            for m in chk['nn']:
                try:
                    self.nn[m].module.load_state_dict(chk['nn'][m])
                except:
                    self.nn[m].load_state_dict(chk['nn'][m])

            for m in chk['optimizer']:
                try:
                    self.optimizer[m].module.load_state_dict(chk['optimizer'][m])
                except:
                    self.optimizer[m].load_state_dict(chk['optimizer'][m])
        else:
            mkey = list(self.nn.keys())[0]
            try:
                self.nn[mkey].module.load_state_dict(chk)
            except:
                self.nn[mkey].load_state_dict(chk)

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
        # self.cache['predictions'] = predictions
        # save_logs(self.cache, file_keys=['predictions'], log_dir=self.cache['log_dir'])
        pass

    def evaluation(self, dataset_list=None, save_pred=False):
        r"""
        Evaluation phase that handles validation/test phase
        split-key: the key to list of files used in this particular evaluation.
        The program will create k-splits(json files) as per specified in --nf -num_of_folds
         argument with keys 'train', ''validation', and 'test'.
        """
        for k in self.nn:
            self.nn[k].eval()

        eval_loss = self.new_averages()
        eval_metrics = self.new_metrics()
        val_loaders = [_dt.COINNDataLoader.new(shuffle=False, dataset=d, **self.cache) for d in dataset_list]
        with _torch.no_grad():
            for loader in val_loaders:
                its = []
                metrics = self.new_metrics()
                for i, batch in enumerate(loader):

                    it = self.iteration(batch)
                    if not it.get('metrics'):
                        it['metrics'] = _base_metrics.ETMetrics()

                    metrics.accumulate(it['metrics'])
                    eval_loss.accumulate(it['averages'])
                    if save_pred:
                        its.append(it)
                eval_metrics.accumulate(metrics)
                if save_pred:
                    self.save_predictions(loader.dataset, its)

        return eval_loss, eval_metrics

    def train_n_eval(self, dataset_cls, nxt_phase):
        out = {'mode': self.input['global_modes'].get(self.state['clientId'], self.cache['mode'])}
        self._load_checkpoint(name=self.cache['current_nn_state'])

        for k in self.nn:
            self.nn[k].train()

        for k in self.optimizer:
            self.optimizer[k].zero_grad()

        if self.input.get('save_current_as_best'):
            self.save_checkpoint(name=self.cache['best_nn_state'])
            self.cache['best_epoch'] = self.cache['epoch']

        if any(m == 'train' for m in self.input['global_modes'].values()):
            """
            All sites must begin/resume the training the same time.
            To enforce this, we have a 'val_waiting' status. Lagged sites will go to this status, 
                and reshuffle the data,
             take part in the training with everybody until all sites go to 'val_waiting' status.
            """
            dataset = dataset_cls(cache=self.cache, state=self.state, mode='train')
            it = self.iteration(dataset.next_batch(dataset, self.cache, mode='train'))
            out.update(**self.backward(it))
            out.update(**self.next_iter())

        if self.input.get('avg_grads'):
            self.step()
            self.save_checkpoint(name=self.cache['current_nn_state'])

        if out['mode'] == 'validation':
            """
            Once all sites are in 'val_waiting' status, remote issues 'validation' signal. 
            Once all sites run validation phase, they go to 'train_waiting' status. 
            Once all sites are in this status, remote issues 'train' signal
             and all sites reshuffle the indices and resume training.
            We send the confusion matrix to the remote to accumulate global score for model selection.
            """
            with open(self.cache['split_dir'] + _sep + self.cache['split_file']) as split_file:
                split = _json.loads(split_file.read())
                val_dataset = dataset_cls(cache=self.cache, state=self.state, mode='eval')
                val_dataset.load_indices()
                val_dataset.load_indices(files=split['validation'])
                avg, scores = self.evaluation([val_dataset], save_pred=False)
                out['validation_log'] = [vars(avg), vars(scores)]
                out.update(**self.next_epoch())

        elif out['mode'] == 'test':
            with open(self.cache['split_dir'] + _sep + self.cache['split_file']) as split_file:
                split = _json.loads(split_file.read())
                self._load_checkpoint(name=self.cache['best_nn_state'])
                test_dataset = dataset_cls(cache=self.cache, state=self.state, mode='eval')
                test_dataset.load_indices()
                test_dataset.load_indices(files=split['validation'])
                avg, scores = self.evaluation([test_dataset], save_pred=True)
                out['test_log'] = [vars(avg), vars(scores)]
                out['mode'] = self.cache['_mode_']
                nxt_phase = 'next_run_waiting'
        return out, nxt_phase
