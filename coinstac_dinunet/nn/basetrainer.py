"""
@author: Aashis Khanal
@email: sraashis@gmail.com
"""

from collections import OrderedDict as _ODict

import torch as _torch

import coinstac_dinunet.config as _conf
import coinstac_dinunet.metrics as _base_metrics
import coinstac_dinunet.utils as _utils
import coinstac_dinunet.utils.tensorutils as _tu
import coinstac_dinunet.vision.plotter as _plot
from coinstac_dinunet.config.keys import *
from coinstac_dinunet.utils import stop_training_
from coinstac_dinunet.utils.logger import *


class NNTrainer:
    def __init__(self, data_handle=None, **kw):
        self.cache = data_handle.cache
        self.input = _utils.FrozenDict(data_handle.input)
        self.state = _utils.FrozenDict(data_handle.state)
        self.nn = _ODict()
        self.device = _ODict()
        self.optimizer = _ODict()
        self.data_handle = data_handle

    def _init_nn_model(self):
        r"""
        User cam override and initialize required models in self.distrib dict.
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

    def init_nn(self, init_model=True, init_optim=True, set_devices=True, init_weights=True):
        if init_model: self._init_nn_model()
        if init_optim: self._init_optimizer()
        if init_weights: self._init_nn_weights(init_weights=init_weights)
        if set_devices: self._set_gpus()

    def _set_gpus(self):
        self.device['gpu'] = _torch.device("cpu")
        if self.cache.get('gpus') is not None and len(self.cache['gpus']) > 0:
            if _conf.gpu_available:
                self.device['gpu'] = _torch.device(f"cuda:{self.cache['gpus'][0]}")
                if len(self.cache['gpus']) >= 2:
                    for mkey in self.nn:
                        self.nn[mkey] = _torch.nn.DataParallel(self.nn[mkey], self.cache['gpus'])
            else:
                pass
                # raise Exception(f'*** GPU not detected in {self.state["clientId"]}. ***')
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

    def evaluation(self, mode='eval', dataset_list=None, save_pred=False):
        for k in self.nn:
            self.nn[k].eval()

        eval_avg, eval_metrics = self.new_averages(), self.new_metrics()
        eval_loaders = []

        for d in dataset_list:
            bz = _tu.get_safe_batch_size(self.cache['batch_size'], len(d))
            eval_loaders.append(
                self.data_handle.get_loader(handle_key=mode, batch_size=bz, dataset=d, shuffle=False)
            )

        with _torch.no_grad():
            for loader in eval_loaders:
                its = []
                metrics = self.new_metrics()
                avg = self.new_averages()
                for i, batch in enumerate(loader, 1):
                    it = self.iteration(batch)
                    metrics.accumulate(it['metrics']), avg.accumulate(it['averages'])

                    if save_pred:
                        its.append(it)

                    if self.cache.get('verbose') and len(dataset_list) <= 1 and lazy_debug(i):
                        info(f" Itr:{i}/{len(loader)}, {it['averages'].get()}, {it['metrics'].get()}")

                eval_metrics.accumulate(metrics)
                eval_avg.accumulate(avg)
                if self.cache.get('verbose') and len(dataset_list) > 1:
                    info(f"{mode} metrics: {avg.get()}, {metrics.get()}")
                if save_pred:
                    self.save_predictions(loader.dataset, self.reduce_iteration(its))

            info(f"{mode} metrics: {eval_avg.get()}, {eval_metrics.get()}", self.cache.get('verbose'))
            return eval_avg, eval_metrics

    def training_iteration_local(self, i, batch):
        r"""
        Learning step for one batch.
        We decoupled it so that user could implement any complex/multi/alternate training strategies.
        """
        it = self.iteration(batch)
        it['loss'].backward()
        if i % self.cache.get('local_iterations', 1) == 0:
            first_optim = list(self.optimizer.keys())[0]
            self.optimizer[first_optim].step()
            self.optimizer[first_optim].zero_grad()
        return it

    def init_training_cache(self):
        self._set_monitor_metric()
        self._set_log_headers()
        self.cache[Key.TRAIN_LOG] = []
        self.cache[Key.VALIDATION_LOG] = []
        metric_direction = self.cache['monitor_metric'][1]
        self.cache['best_val_epoch'] = 0
        self.cache.update(best_val_score=0.0 if metric_direction == 'maximize' else _conf.max_size)

    def train_local(self, train_dataset, val_dataset):
        out = {}

        if not isinstance(val_dataset, list):
            val_dataset = [val_dataset]

        if self.data_handle.dataloader_args.get('drop_last') or self.cache.get('drop_last'):
            bz = self.cache['batch_size']
        else:
            bz = _tu.get_safe_batch_size(self.cache['batch_size'], len(train_dataset))

        loader = self.data_handle.get_loader('train', dataset=train_dataset, shuffle=True, batch_size=bz)
        local_iter = self.cache.get('local_iterations', 1)
        tot_iter = len(loader) // local_iter
        for ep in range(1, self.cache['epochs'] + 1):
            for k in self.nn:
                self.nn[k].train()

            _metrics, _avg = self.new_metrics(), self.new_averages()
            ep_avg, ep_metrics, its = self.new_averages(), self.new_metrics(), []

            for i, batch in enumerate(loader, 1):
                its.append(self.training_iteration_local(i, batch))
                if i % local_iter == 0:
                    it = self.reduce_iteration(its)

                    ep_avg.accumulate(it['averages']), ep_metrics.accumulate(it['metrics'])
                    _avg.accumulate(it['averages']), _metrics.accumulate(it['metrics'])

                    _i, its = i // local_iter, []
                    if lazy_debug(_i) or _i == tot_iter:
                        info(f"Ep:{ep}/{self.cache['epochs']},Itr:{_i}/{tot_iter},{_avg.get()},{_metrics.get()}",
                             self.cache.get('verbose'))
                        self.cache[Key.TRAIN_LOG].append([*_avg.get(), *_metrics.get()])
                        _metrics.reset(), _avg.reset()
                    self.on_iteration_end(i=_i, ep=ep, it=it)

            if ep % self.cache.get('validation_epochs', 1) == 0:
                val_averages, val_metric = self.evaluation(mode='validation', dataset_list=val_dataset)
                self.cache[Key.VALIDATION_LOG].append([*val_averages.get(), *val_metric.get()])
                out.update(**self._save_if_better(ep, val_metric))

                self._on_epoch_end(ep=ep, ep_averages=ep_avg, ep_metrics=ep_metrics,
                                   val_averages=val_averages, val_metrics=val_metric)

                if lazy_debug(ep):
                    self._save_progress(self.cache, epoch=ep)

                if self._stop_early(ep, val_metric, val_averages=val_averages,
                                    epoch_averages=ep_avg, epoch_metrics=ep_metrics):
                    break

        self._save_progress(self.cache, epoch=ep)
        _utils.save_cache(self.cache, self.cache['log_dir'])
        return out

    def iteration(self, batch):
        r"""
        Left for user to implement one mini-bath iteration:
        Example:{
                    inputs = batch['input'].to(self.device['gpu']).float()
                    labels = batch['label'].to(self.device['gpu']).long()
                    out = self.distrib['model'](inputs)
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
        return {}

    def save_predictions(self, dataset, its):
        pass

    def reduce_iteration(self, its):
        reduced = {}.fromkeys(its[0].keys(), None)
        for key in reduced:
            if isinstance(its[0][key], _base_metrics.COINNAverages):
                reduced[key] = self.new_averages()
                [reduced[key].accumulate(ik[key]) for ik in its]

            elif isinstance(its[0][key], _base_metrics.COINNMetrics):
                reduced[key] = self.new_metrics()
                [reduced[key].accumulate(ik[key]) for ik in its]
            else:
                def collect(k=key, src=its):
                    _data = []
                    is_tensor = isinstance(src[0][k], _torch.Tensor)
                    is_tensor = is_tensor and not src[0][k].requires_grad and src[0][k].is_leaf
                    for ik in src:
                        if is_tensor:
                            _data.append(ik[k] if len(ik[k].shape) > 0 else ik[k].unsqueeze(0))
                        else:
                            _data.append(ik[k])
                    if is_tensor:
                        return _torch.cat(_data)
                    return _data

                reduced[key] = collect

        return reduced

    def _set_monitor_metric(self):
        self.cache['monitor_metric'] = 'time', 'maximize'

    def _save_if_better(self, epoch, val_metrics):
        return {}

    def new_metrics(self):
        return _base_metrics.COINNMetrics()

    def new_averages(self):
        return _base_metrics.COINNAverages(num_averages=1)

    def _on_epoch_end(self, ep, **kw):
        r"""
        Any logic to run after an epoch ends.
        """
        return {}

    def on_iteration_end(self, i, ep, it):
        r"""
        Any logic to run after an iteration ends.
        """
        return {}

    def _save_progress(self, cache, epoch):
        _plot.plot_progress(cache, self.cache['log_dir'], plot_keys=[Key.TRAIN_LOG], epoch=epoch)
        _plot.plot_progress(cache, self.cache['log_dir'], plot_keys=[Key.VALIDATION_LOG],
                            epoch=epoch // self.cache['validation_epochs'])

    def _stop_early(self, epoch, val_metrics=None, **kw):
        return stop_training_(epoch, self.cache)

    def _set_log_headers(self):
        self.cache['log_header'] = 'Loss,Accuracy'
