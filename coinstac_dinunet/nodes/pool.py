from coinstac_dinunet import COINNTrainer as _BaseTrainer
import json as _json
import os as _os
import coinstac_dinunet.config as _conf


def PooledTrainer(base=_BaseTrainer, dataset_dir='test', log_dir='net_logs', **kw):
    class COINNPooledTrainer(base):
        def __init__(self, dataset_dir=dataset_dir, log_dir=log_dir, **kw):
            self.dataset_dir = dataset_dir
            self.log_dir = log_dir
            self.inputspecs = self.parse_inputspec(dataset_dir + _os.sep + kw.get('inputspec_file', 'inputspec.json'))

            cache = {**self.inputspecs[0], 'folds': self.init_folds()}
            cache.update(**kw)
            super().__init__(cache=cache, input={}, state={}, **kw)

        def init_folds(self):
            folds = {}
            for site, inputspec in self.inputspecs.items():
                folds[site] = sorted(_os.listdir(self.base_directory(site) + _os.sep + inputspec['split_dir']))
            return folds

        def parse_inputspec(self, inputspec_path):
            inputspec = {}
            for site, isp in enumerate(_json.loads(open(inputspec_path).read())):
                spec = {}
                for k, v in isp.items():
                    spec[k] = v['value']
                inputspec[site] = spec
            return inputspec

        def _get_train_dataset(self, dataset_cls):
            return self._load_dataset(dataset_cls, 'train')

        def _load_dataset(self, dataset_cls, key):
            dataset = dataset_cls(mode='pre_train', limit=self.cache.get('load_limit', _conf.data_load_lim))
            for site, fold in self.cache['folds'].items():
                split = fold[self.cache['fold_ix']]
                path = self.base_directory(site) + _os.sep + self.inputspecs[site]['split_dir']
                split = _json.loads(open(path + _os.sep + split).read())
                dataset.add(files=split['train'],
                            cache={'args': self.inputspecs[site]},
                            state={'clientId': site, "baseDirectory": self.base_directory(site)})
            return dataset

        def _save_if_better(self, epoch, metrics):
            monitor_metric, direction = self.cache['monitor_metric']
            sc = getattr(metrics, monitor_metric)
            if callable(sc):
                sc = sc()
            if (direction == 'maximize' and sc >= self.cache['best_local_score']) or (
                    direction == 'minimize' and sc <= self.cache['best_local_score']):
                self.cache['best_local_epoch'] = epoch
                self.cache['best_local_score'] = sc
                self.save_checkpoint(file_path=self.cache['log_dir'] + _os.sep + _conf.weights_file)
            if self.cache.get('verbose'):
                print(f"--- ### Best Model Saved!!! --- : {self.cache['best_local_score']}")
            else:
                if self.cache.get('verbose'):
                    print(f"Not best!  {sc}, {self.cache['best_local_score']} in ep: {self.cache['best_local_epoch']}")
            return {}

        def base_directory(self, site):
            return f"{self.dataset_dir}/input/local{site}/simulatorRun"

        def run(self, dataset_cls):
            for fold_ix in range(len(self.cache['folds'][0])):
                self.cache['fold_ix'] = fold_ix
                self.cache['log_dir'] = self.log_dir + _os.sep + f'fold_{fold_ix}'
                self.cache['args'] = {**self.cache}
                _os.makedirs(self.cache['log_dir'], exist_ok=True)
                self.train_local(dataset_cls, verbose=True)

    trainer = COINNPooledTrainer(dataset_dir=dataset_dir, log_dir=log_dir, **kw)
    trainer.init_nn(True)
    return trainer
