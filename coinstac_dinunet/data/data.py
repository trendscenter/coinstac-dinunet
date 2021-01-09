"""
@author: Aashis Khanal
@email: sraashis@gmail.com
@ref: https://github.com/sraashis/easytorch
"""

import json as _json
from os import sep as _sep

from torch.utils.data import Dataset as _Dataset, DataLoader as _DataLoader
from torch.utils.data._utils.collate import default_collate as _default_collate


def safe_collate(batch):
    return _default_collate([b for b in batch if b])


class COINNDataLoader(_DataLoader):

    def __init__(self, **kw):
        super(COINNDataLoader, self).__init__(**kw)

    @classmethod
    def new(cls, **kw):
        _kw = {
            'data': None,
            'batch_size': 1,
            'shuffle': False,
            'sampler': None,
            'batch_sampler': None,
            'num_workers': 0,
            'pin_memory': False,
            'drop_last': False,
            'timeout': 0,
            'worker_init_fn': None
        }
        for k in _kw.keys():
            _kw[k] = kw.get(k, _kw.get(k))
        return cls(collate_fn=safe_collate, **_kw)


class COINNDataset(_Dataset):
    def __init__(self, cache={}, state={}, mode=None, **kw):
        self.data_dir = state.get('baseDirectory', '') + _sep + cache.get('data_dir', '')
        self.label_dir = state.get('baseDirectory', '') + _sep + cache.get('label_dir', '')
        self.mode = mode
        self.indices = kw.get('indices', [])

    def load_indices(self, **kw):
        return NotImplementedError('Must be implemented.')

    def __getitem__(self, ix):
        return NotImplementedError('Must be implemented.')

    def __len__(self):
        return len(self.indices)

    def loader(self, shuffle=False, batch_size=None, num_workers=0, pin_memory=True, **kw):
        return COINNDataLoader.new(dataset=self, shuffle=shuffle, batch_size=batch_size,
                                   num_workers=num_workers, pin_memory=pin_memory, **kw)

    def cache_data_indices(self, cache, min_batch_size=4):
        """
        Parse and load dataset and save to cache:
        so that in next global iteration we dont have to do that again.
        The data IO depends on use case-For a instance, if your data can fit in RAM, you can load
         and save the entire dataset in cache. But, in general,
         it is better to save indices in cache, and load only a mini-batch at a time
         (logic in __nextitem__) of the data loader.
        """
        split = _json.loads(open(cache['split_dir'] + _sep + cache['split_file']).read())
        self.load_indices(files=split['train'])

        cache['data_indices'] = self.indices
        if len(self) % cache['batch_size'] >= min_batch_size:
            cache['data_len'] = len(self)
        else:
            cache['data_len'] = (len(self) // cache['batch_size']) * cache['batch_size']

    def next_batch(self, cache):
        self.indices = cache['data_indices'][cache['cursor']:]
        loader = self.loader(batch_size=cache['batch_size'], num_workers=cache.get('num_workers', 0),
                             pin_memory=cache.get('pin_memory', True))
        return next(loader.__iter__())


def create_ratio_split(cache, shuffle_files=True, name='SPLIT', ):
    files = os.listdir(state['baseDirectory'] + sep + cache['data_dir'])
    save_to_dir = cache['split_dir']
    ratio = cache.get('split_ratio', (0.6, 0.2, 0.2))
    first_key = cache.get('first_key', 'train')

    if shuffle_files:
        shuffle(files)

    keys = [first_key]
    if len(ratio) == 2:
        keys.append('test')
    elif len(ratio) == 3:
        keys.append('validation')
        keys.append('test')

    _ratio = ratio[::-1]
    locs = _np.array([sum(_ratio[0:i + 1]) for i in range(len(ratio) - 1)])
    locs = (locs * len(files)).astype(int)
    splits = _np.split(files[::-1], locs)[::-1]
    splits = dict([(k, sp.tolist()[::-1]) for k, sp in zip(keys, splits)])
    if save_to_dir:
        f = open(save_to_dir + _sep + f'{name}.json', "w")
        f.write(_json.dumps(splits))
        f.close()
    else:
        return splits


def create_k_fold_splits(cache, shuffle_files=True, name='SPLIT'):
    from random import shuffle
    import numpy as np

    files = os.listdir(state['baseDirectory'] + sep + cache['data_dir'])
    k = cache['num_folds']
    save_to_dir = cache['split_dir']

    if shuffle_files:
        shuffle(files)

    ix_splits = np.array_split(np.arange(len(files)), k)
    for i in range(len(ix_splits)):
        test_ix = ix_splits[i].tolist()
        val_ix = ix_splits[(i + 1) % len(ix_splits)].tolist()
        train_ix = [ix for ix in np.arange(len(files)) if ix not in test_ix + val_ix]

        splits = {'train': [files[ix] for ix in train_ix],
                  'validation': [files[ix] for ix in val_ix],
                  'test': [files[ix] for ix in test_ix]}

        if save_to_dir:
            f = open(save_to_dir + os.sep + f'{name}_' + str(i) + '.json', "w")
            f.write(json.dumps(splits))
            f.close()
        else:
            return splits


def init_k_folds(cache, state, data_splitter=create_k_fold_splits):
    """
    If one wants to use custom splits:- Populate splits_dir as specified in inputs spec with split files(.json)
        with list of file names on each train, validation, and test keys.
    Number of split files should be equal to num_folds passed in inputspec
    If nothing is provided, random k-splits will be created.
    Splits will be copied/created in output directory to have everything of a result at the same place.
    """
    out = {}
    cache['split_dir'] = cache.get('split_dir', 'splits')
    split_dir = state['baseDirectory'] + sep + cache['split_dir']

    cache['split_dir'] = state['outputDirectory'] + sep + cache['id'] + sep + cache['split_dir']
    shutil.rmtree(cache['split_dir'], ignore_errors=True)
    os.makedirs(cache['split_dir'], exist_ok=True)

    if not os.path.exists(split_dir) or len(os.listdir(split_dir)) == 0:
        data_splitter(cache)

    elif cache.get('num_folds') and len(os.listdir(split_dir)) != cache['num_folds']:
        raise ValueError(f"Number of splits in {split_dir} of site {state['clientId']} \
                                must be {cache['num_folds']} instead of {len(os.listdir(split_dir))}")

    [shutil.copy(split_dir + sep + f, cache['split_dir'] + sep + f) for f in os.listdir(split_dir)]
    splits = sorted(os.listdir(cache['split_dir']))
    cache['splits'] = dict(zip(range(len(splits)), splits))
    out['num_folds'] = cache['num_folds']
    out['id'] = cache['id']
    out['seed'] = cache.get('seed')
    return out
