import json as _json
import os as _os
import random as _rd
import shutil as _shutil

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
        f = open(save_to_dir + _os.sep + f'{name}.json', "w")
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
            f = open(save_to_dir + _os.sep + f'{name}_' + str(i) + '.json', "w")
            f.write(_json.dumps(splits))
            f.close()
        else:
            return splits


def split_place_holder(files, cache):
    save_to_dir = cache['split_dir']
    splits = {'train': [], 'validation': [], 'test': []}
    f = open(save_to_dir + _os.sep + f'empty_split.json', "w")
    f.write(_json.dumps(splits))


def init_k_folds(files, cache, state):
    """
    If one wants to use custom splits:- Populate splits_dir as specified in inputs spec with split files(.json)
        with list of file names on each train, validation, and test keys.
    Number of split files should be equal to num_folds passed in inputspec
    If nothing is provided, random k-splits will be created.
    Splits will be copied/created in output directory to have everything of a result at the same place.
    """

    data_splitter = split_place_holder
    if cache.get('split_ratio') is not None:
        data_splitter = create_ratio_split
    elif cache.get('num_folds') is not None:
        data_splitter = create_k_fold_splits

    out = {}
    cache['split_dir'] = cache.get('split_dir', 'splits')
    split_dir = state['baseDirectory'] + _os.sep + cache['split_dir']

    cache['split_dir'] = state['outputDirectory'] + _os.sep + cache['task_id'] + _os.sep + cache['split_dir']
    _os.makedirs(cache['split_dir'], exist_ok=True)

    if _os.path.exists(split_dir) and len(_os.listdir(split_dir)) > 0:
        [_shutil.copy(split_dir + _os.sep + f, cache['split_dir'] + _os.sep + f) for f in _os.listdir(split_dir)]

    elif len(_os.listdir(cache['split_dir'])) == 0:
        data_splitter(files, cache)

    splits = sorted(_os.listdir(cache['split_dir']))
    cache['splits'] = dict(zip([str(i) for i in range(len(splits))], splits))
    return out
