import os as _os
import json as _json
import copy as _copy
import math as _math
from .utils import *


class FrozenDict(dict):
    def __init__(self, _dict):
        super().__init__(_dict)

    def prompt(self, key, value):
        raise ValueError(f'*** '
                         f'Attempt to modify frozen dict '
                         f'[{key} : {self[key]}] with [{key} : {value}]'
                         f' ***')

    def __setitem__(self, key, value):
        if key not in self:
            super(FrozenDict, self).__setitem__(key, value)
        else:
            self.prompt(key, value)

    def update(self, **kw):
        for k, v in kw.items():
            self[k] = v


def save_scores(cache, log_dir, file_keys=[]):
    for fk in file_keys:
        with open(log_dir + _os.sep + f'{fk}.csv', 'w') as file:
            header = cache.get('log_header', '')
            if isinstance(header, list):
                header = ','.join(header)
            file.write(header + '\n')
            for line in cache[fk]:
                if isinstance(line, list):
                    file.write(','.join([str(s) for s in line]) + '\n')
                else:
                    file.write(f'{line}\n')


def jsonable(obj):
    try:
        _json.dumps(obj)
        return True
    except:
        return False


def clean_recursive(obj):
    r"""
    Make everything in cache safe to save in json files.
    """
    if not isinstance(obj, dict):
        return
    for k, v in obj.items():
        if isinstance(v, dict):
            clean_recursive(v)
        elif isinstance(v, list):
            for i in v:
                clean_recursive(i)
        elif not jsonable(v):
            obj[k] = f'{v}'


def save_cache(cache, log_dir):
    with open(log_dir + _os.sep + f"logs.json", 'w') as fp:
        log = {}.fromkeys(cache.keys())
        for k in log:
            try:
                log[k] = _copy.deepcopy(cache[k])
            except:
                log[k] = ''
        clean_recursive(log)
        _json.dump(log, fp)


def lazy_debug(x, add=1):
    return x % int(_math.log(x + 1) + add) == 0
