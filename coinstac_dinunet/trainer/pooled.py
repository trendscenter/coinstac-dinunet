"""
@author: Aashis Khanal
@email: sraashis@gmail.com
@ref: https://github.com/sraashis/easytorch
"""

import json as _json
import os as _os

import coinstac_dinunet.config as _conf
from coinstac_dinunet import utils as _utils
from coinstac_dinunet.utils.logger import *
from coinstac_dinunet.config.status import *
from .nn import NNTrainer as _NNTrainer


def PooledTrainer(base=_NNTrainer, dataset_dir='test', log_dir='net_logs', **kw):
    class PooledTrainer(base):
        def __init__(self, dataset_dir=dataset_dir, log_dir=log_dir, **kw):
            self.dataset_dir = dataset_dir
            self.log_dir = log_dir
            self.inputspecs = self._parse_inputspec(dataset_dir + _os.sep + kw.get('inputspec_file', 'inputspec.json'))

            cache = {**self.inputspecs[0],
                     'folds': self._init_folds(),
                     'seed': _conf.current_seed}
            cache.update(**kw)
            super().__init__(cache=cache, input={}, state={}, **kw)

        def _init_folds(self):
            folds = {}
            for site, inputspec in self.inputspecs.items():
                folds[site] = sorted(_os.listdir(self.base_directory(site) + _os.sep + inputspec['split_dir']))
            return folds

        def _parse_inputspec(self, inputspec_path):
            inputspec = {}
            for site, isp in enumerate(_json.loads(open(inputspec_path).read())):
                spec = {}
                for k, v in isp.items():
                    spec[k] = v['value']
                inputspec[site] = spec
            return inputspec

        def _load_dataset(self, dataset_cls, split_key):
            dataset = dataset_cls(mode='pre_train', limit=self.cache.get('load_limit', _conf.max_size))
            for site, fold in self.cache['folds'].items():
                split = fold[self.cache['fold_ix']]
                path = self.base_directory(site) + _os.sep + self.inputspecs[site]['split_dir']
                split = _json.loads(open(path + _os.sep + split).read())
                dataset.add(files=split[split_key],
                            cache={'args': self.inputspecs[site]},
                            state={'clientId': site, "baseDirectory": self.base_directory(site)})
            return dataset

        def _save_if_better(self, epoch, metrics):
            monitor_metric, direction = self.cache['monitor_metric']
            sc = getattr(metrics, monitor_metric)
            if callable(sc):
                sc = sc()
            if (direction == 'maximize' and sc > self.cache['best_local_score']) or (
                    direction == 'minimize' and sc < self.cache['best_local_score']):
                self.cache['best_local_epoch'] = epoch
                self.cache['best_local_score'] = sc
                self.save_checkpoint(file_path=self.cache['log_dir'] + _os.sep + _conf.weights_file)
                success(f"--- ### Best Model Saved!!! --- : {self.cache['best_local_score']}",
                        self.cache.get('verbose'))
            else:
                warn(f"Not best!  {sc}, {self.cache['best_local_score']} in ep: {self.cache['best_local_epoch']}",
                     self.cache.get('verbose'))
            return {}

        def base_directory(self, site):
            return f"{self.dataset_dir}/input/local{site}/simulatorRun"

        def run(self, dataset_cls, only_sites: list = None, only_folds: list = None):
            self.init_nn(True)
            global_avg, global_metrics = self.new_averages(), self.new_metrics()

            if only_sites is not None:
                self.enable_sites(only_sites)

            first_site = list(self.cache['folds'].keys())[0]

            folds = only_folds
            if folds is None:
                folds = range(len(self.cache['folds'][first_site]))

            for fold_ix in folds:
                self.cache['fold_ix'] = fold_ix
                self.cache['log_dir'] = self.log_dir + _os.sep + f'fold_{fold_ix}'
                self.cache['args'] = {**self.cache}
                _os.makedirs(self.cache['log_dir'], exist_ok=True)

                if self.cache['mode'] == Mode.TRAIN:
                    self.train_local(dataset_cls)

                if self.cache['mode'] == 'train' or self.cache['pretrained_path'] is None:
                    self.load_checkpoint(self.cache['log_dir'] + _os.sep + _conf.weights_file)
                test_datasets = self._load_dataset(dataset_cls, 'test')
                test_averages, test_metrics = self.evaluation(mode='test', dataset_list=[test_datasets], save_pred=True)

                global_avg.accumulate(test_averages), global_metrics.accumulate(test_metrics)
                self.cache['test_score'] = [[*test_averages.get(), *test_metrics.get()]]
                info(f"Fold {fold_ix}, {self.cache['test_score'][0]}", self.cache.get('verbose'))
                _utils.save_scores(self.cache, log_dir=self.cache['log_dir'], file_keys=['test_score'])
                _utils.save_cache(self.cache, log_dir=self.cache['log_dir'])

            self.cache[Key.GLOBAL_TEST_METRICS] = [[*global_avg.get(), *global_metrics.get()]]
            success(f"Global: {self.cache[Key.GLOBAL_TEST_METRICS]}", self.cache.get('verbose'))
            _utils.save_scores(self.cache, log_dir=self.log_dir, file_keys=[Key.GLOBAL_TEST_METRICS])

        def enable_sites(self, sites: list = None):
            self.inputspecs = {site: self.inputspecs[site] for site in sites}
            self.cache['folds'] = {site: self.cache['folds'][site] for site in sites}

    return PooledTrainer(dataset_dir=dataset_dir, log_dir=log_dir, verbose=True, **kw)
