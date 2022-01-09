import json as _json
import os as _os
import random as _rd
import shutil as _shutil

_sep = _os.sep

import numpy as _np


def create_ratio_split(files, cache, shuffle_files=True, name='SPLIT'):
    save_to_dir = cache['split_dir']
    ratio = cache.get('split_ratio', (0.6, 0.2, 0.2))
    first_key = cache.get('first_key', 'train')

    if shuffle_files:
        _rd.shuffle(files)

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


def create_k_fold_splits(files, cache, shuffle_files=True, name='SPLIT'):
    k = cache['num_folds']
    save_to_dir = cache['split_dir']
    if shuffle_files:
        _rd.shuffle(files)
    file_ix = _np.arange(len(files))
    ix_splits = _np.array_split(file_ix, k)
    for i in range(len(ix_splits)):
        test_ix = ix_splits[i].tolist()
        val_ix = ix_splits[(i + 1) % len(ix_splits)].tolist()
        train_ix = _np.delete(file_ix.copy(), _np.array(test_ix + val_ix))

        splits = {"train": [files[ix] for ix in train_ix],
                  "validation": [files[ix] for ix in val_ix],
                  "test": [files[ix] for ix in test_ix]}

        if save_to_dir:
            f = open(save_to_dir + _sep + f'{name}_' + str(i) + '.json', "w")
            f.write(_json.dumps(splits))
            f.close()
        else:
            return splits


def split_place_holder(files, cache):
    save_to_dir = cache['split_dir']
    splits = {'train': [], 'validation': [], 'test': []}
    f = open(save_to_dir + _sep + f'empty_split.json', "w")
    f.write(_json.dumps(splits))


def init_k_folds(files, cache, state):
    out = {}

    """Splits dir path given"""
    _dir = state['baseDirectory'] + _sep + cache.get('split_dir', 'splits')

    cache['split_dir'] = state['outputDirectory'] + _sep + cache['task_id'] + _sep + 'splits'
    _os.makedirs(cache['split_dir'], exist_ok=True)
    if _os.path.exists(_dir) and len(_os.listdir(_dir)) > 0:
        [_shutil.copy(_dir + _sep + f, cache['split_dir'] + _sep + f) for f in _os.listdir(_dir)]

    elif cache.get('split_files'):
        [_shutil.copy(state['baseDirectory'] + _sep + f, cache['split_dir'] + _sep + f) for f in cache['split_files']]
    
    elif cache.get('num_folds'):
        create_k_fold_splits(files, cache)

    elif cache.get('split_ratio'):
        create_ratio_split(files, cache)

    else:
        split_place_holder(None, cache)

    splits = sorted(_os.listdir(cache['split_dir']))
    cache['splits'] = dict(zip([str(i) for i in range(len(splits))], splits))
    return out
