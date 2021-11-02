import datetime as _dt
import time as _t

from coinstac_dinunet import config as _conf


def performance_improved_(epoch, score, cache):
    delta = cache.get('score_delta', _conf.score_delta)
    improved = False
    if cache['metric_direction'] == 'maximize':
        improved = score > cache['best_val_score'] + delta
    elif cache['metric_direction'] == 'minimize':
        improved = score < cache['best_val_score'] - delta

    if improved:
        cache['best_val_epoch'] = epoch
        cache['best_val_score'] = score
    return bool(improved)


def stop_training_(epoch, cache):
    return epoch - cache['best_val_epoch'] > cache.get('patience', cache['epochs'])


def duration(cache: dict, begin, key):
    t_del = _dt.datetime.fromtimestamp(_t.time()) - _dt.datetime.fromtimestamp(begin)
    if cache.get(key) is None:
        cache[key] = [t_del.total_seconds()]
    else:
        cache[key].append(t_del.total_seconds())
    return t_del
