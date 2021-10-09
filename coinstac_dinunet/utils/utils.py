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
    if epoch - cache['best_val_epoch'] > cache.get('patience', cache['epochs']):
        return True

    if cache['metric_direction'] == 'maximize':
        return cache['best_val_score'] == _conf.score_high
    elif cache['metric_direction'] == 'minimize':
        return cache['best_val_score'] == _conf.score_low
    return False


def duration(cache: dict, begin, key):
    t_del = _dt.datetime.fromtimestamp(_t.time()) - _dt.datetime.fromtimestamp(begin)
    if cache.get(key) is None:
        cache[key] = t_del
    else:
        cache[key] += t_del
    return t_del
