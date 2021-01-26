import json as _json
import os as _os
import random as _rd
import shutil as _shutil

import numpy as _np


def create_ratio_split(cache, state, shuffle_files=True, name='SPLIT'):
    files = _os.listdir(state['baseDirectory'] + _os.sep + cache['data_dir'])
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


def create_k_fold_splits(cache, state, shuffle_files=True, name='SPLIT'):
    files = _os.listdir(state['baseDirectory'] + _os.sep + cache['data_dir'])
    k = cache['num_folds']
    save_to_dir = cache['split_dir']
    if shuffle_files:
        _rd.shuffle(files)
    ix_splits = _np.array_split(_np.arange(len(files)), k)
    for i in range(len(ix_splits)):
        test_ix = ix_splits[i].tolist()
        val_ix = ix_splits[(i + 1) % len(ix_splits)].tolist()
        train_ix = [ix for ix in _np.arange(len(files)) if ix not in test_ix + val_ix]

        splits = {'train': [files[ix] for ix in train_ix],
                  'validation': [files[ix] for ix in val_ix],
                  'test': [files[ix] for ix in test_ix]}

        if save_to_dir:
            f = open(save_to_dir + _os.sep + f'{name}_' + str(i) + '.json', "w")
            f.write(_json.dumps(splits))
            f.close()
        else:
            return splits


def init_k_folds(cache, state, data_splitter=None):
    """
    If one wants to use custom splits:- Populate splits_dir as specified in inputs spec with split files(.json)
        with list of file names on each train, validation, and test keys.
    Number of split files should be equal to num_folds passed in inputspec
    If nothing is provided, random k-splits will be created.
    Splits will be copied/created in output directory to have everything of a result at the same place.
    """

    if not data_splitter and cache.get('split_ratio') is not None: data_splitter = create_ratio_split
    if not data_splitter and cache.get('num_folds') is not None: data_splitter = create_k_fold_splits

    out = {}
    cache['split_dir'] = cache.get('split_dir', 'splits')
    split_dir = state['baseDirectory'] + _os.sep + cache['split_dir']

    cache['split_dir'] = state['outputDirectory'] + _os.sep + cache['computation_id'] + _os.sep + cache['split_dir']
    _shutil.rmtree(cache['split_dir'], ignore_errors=True)
    _os.makedirs(cache['split_dir'], exist_ok=True)

    if not _os.path.exists(split_dir) or len(_os.listdir(split_dir)) == 0:
        data_splitter(cache, state)

    elif cache.get('num_folds') and len(_os.listdir(split_dir)) != cache['num_folds']:
        raise ValueError(f"Number of splits in {split_dir} of site {state['clientId']} \
                                must be {cache['num_folds']} instead of {len(_os.listdir(split_dir))}")

    [_shutil.copy(split_dir + _os.sep + f, cache['split_dir'] + _os.sep + f) for f in _os.listdir(split_dir)]
    splits = sorted(_os.listdir(cache['split_dir']))

    cache['splits'] = dict(zip([str(i) for i in range(len(splits))], splits))
    out['num_folds'] = len(splits)
    return out