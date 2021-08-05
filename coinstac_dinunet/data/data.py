"""
@author: Aashis Khanal
@email: sraashis@gmail.com
"""
import json as _json
import os as _os

import numpy as _np
from torch.utils.data import DataLoader as _DataLoader, Dataset as _Dataset
from torch.utils.data._utils.collate import default_collate as _default_collate

import coinstac_dinunet.config as _conf
import coinstac_dinunet.utils as _utils
from coinstac_dinunet.config.keys import *
from coinstac_dinunet.utils.logger import *
from .datautils import init_k_folds as _kfolds

_sep = _os.sep
import torch as _torch


def safe_collate(batch):
    r"""
    Savely select batches/skip errors in file loading.
    """
    return _default_collate([b for b in batch if b])


class COINNDataset(_Dataset):
    def __init__(self, mode='init', limit=_conf.max_size):
        self.mode = mode
        self.limit = limit
        self.state = {}
        self.inputspecs = {}
        self.indices = []

    def load_index(self, site, file):
        r"""
        Logic to load indices of a single file.
        -Sometimes one image can have multiple indices like U-net where we have to get multiple patches of images.
        """
        self.indices.append([site, file])

    def _load_indices(self, site, files, **kw):
        r"""
        We load the proper indices/names(whatever is called) of the files in order to prepare minibatches.
        Only load lim numbr of files so that it is easer to debug(Default is infinite, -lim/--load-lim argument).
        """
        for file in files:
            if len(self) >= self.limit:
                break
            self.load_index(site, file)

        if kw.get('verbose', True):
            print(f'{site}, {self.mode}, {len(self)} Indices Loaded')

    def __getitem__(self, index):
        r"""
        Logic to load one file and send to model. The mini-batch generation will be handled by Dataloader.
        Here we just need to write logic to deal with single file.
        """
        raise NotImplementedError('Must be implemented by child class.')

    def __len__(self):
        return len(self.indices)

    def transforms(self, **kw):
        return None

    def path(self, site, inputspec_key, root_dir='baseDirectory'):
        return self.state[site][root_dir] + _os.sep + self.inputspecs[site][inputspec_key]

    def add(self, files, cache: dict = None, state: dict = None):
        self.state[state['clientId']] = state
        self.inputspecs[state['clientId']] = cache['args']
        self._load_indices(site=state['clientId'], files=files, verbose=False)


def _seed_worker(worker_id):
    seed = (int(_torch.initial_seed()) + worker_id) % (2 ** 32 - 1)
    _np.random.seed(seed)


class COINNDataHandle:

    def __init__(self, cache=None, input=None, state=None, dataloader_args=None, **kw):
        if dataloader_args is None:
            dataloader_args = {}

        self.cache = cache
        self.input = input
        self.state = state
        self.dataset = self.cache.get('dataset', {})
        self.dataloader_args = _utils.FrozenDict(dataloader_args)

    def get_dataset(self, handle_key, files, dataset_cls=None):
        dataset = dataset_cls(mode=handle_key, limit=self.cache['load_limit'])
        dataset.add(files=files, cache=self.cache, state=self.state)
        self.dataset[handle_key] = dataset
        return self.dataset[handle_key]

    def get_train_dataset(self, dataset_cls):
        if dataset_cls is None or self.dataloader_args.get('train', {}).get('dataset'):
            return self.dataloader_args.get('train', {}).get('dataset')

        r"""Load the train data from current fold/split."""
        with open(self.cache['split_dir'] + _sep + self.cache['split_file']) as file:
            split = _json.loads(file.read())
            train_dataset = self.get_dataset('train', split.get('train', []), dataset_cls=dataset_cls)
            return train_dataset

    def get_validation_dataset(self, dataset_cls):
        if dataset_cls is None or self.dataloader_args.get('validation', {}).get('dataset'):
            return self.dataloader_args.get('validation', {}).get('dataset')

        r""" Load the validation data from current fold/split."""
        with open(self.cache['split_dir'] + _sep + self.cache['split_file']) as file:
            split = _json.loads(file.read())
            val_dataset = self.get_dataset('validation', split.get('validation', []), dataset_cls=dataset_cls)
            if val_dataset and len(val_dataset) > 0:
                return val_dataset

    def get_test_dataset(self, dataset_cls):
        if dataset_cls is None or self.dataloader_args.get('test', {}).get('dataset'):
            return self.dataloader_args.get('test', {}).get('dataset')

        with open(self.cache['split_dir'] + _sep + self.cache['split_file']) as file:
            _files = _json.loads(file.read()).get('test', [])[:self.cache['load_limit']]
            if self.cache['load_sparse'] and len(_files) > 1:
                datasets = [self.get_dataset('test', [f], dataset_cls=dataset_cls) for f in _files]
                success(f'\n{len(datasets)} sparse dataset loaded.', self.cache['verbose'])
            else:
                datasets = self.get_dataset('test', _files, dataset_cls=dataset_cls)

            if len(datasets) > 0 and sum([len(t) for t in datasets if t]) > 0:
                return datasets

    def get_loader(self, handle_key='', **kw):
        args = {**self.cache}
        args.update(self.dataloader_args.get(handle_key, {}))
        args.update(**kw)

        loader_args = {
            'dataset': None,
            'batch_size': 1,
            'sampler': None,
            'shuffle': False,
            'batch_sampler': None,
            'num_workers': 0,
            'pin_memory': False,
            'drop_last': False,
            'timeout': 0,
            'worker_init_fn': _seed_worker if args.get('seed_all') else None
        }
        for k in loader_args.keys():
            loader_args[k] = args.get(k, loader_args.get(k))

        return _DataLoader(collate_fn=safe_collate, **loader_args)

    def next_iter(self, handle_key=Mode.TRAIN, shuffle=True) -> tuple:
        out = {}

        if self.cache['cursor'] == 0:
            dataset = self.dataset[handle_key]
            loader = self.get_loader(handle_key=handle_key, shuffle=shuffle, dataset=dataset)
            self.cache['data_len'] = (len(dataset) // self.cache['batch_size']) * self.cache['batch_size']
            self.cache['train_loader_iter'] = iter(loader)

        batch = next(self.cache['train_loader_iter'])
        self.cache['cursor'] += self.cache['batch_size']

        if self.cache['cursor'] >= self.cache['data_len']:
            out['mode'] = Mode.VALIDATION_WAITING
            self.cache['cursor'] = 0

        return batch, out

    def prepare_data(self):
        files = []
        if self.cache.get('data_dir'):
            files = _os.listdir(self.state['baseDirectory'] + _os.sep + self.cache['data_dir'])
        return _kfolds(files, self.cache, self.state)
