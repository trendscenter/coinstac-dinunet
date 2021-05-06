from coinstac_dinunet import config as _conf
from coinstac_dinunet.profiler import default_args as _args
import os as _os
import json as _json
import traceback as _tb


def performance_improved_(epoch, score, cache):
    monitor_metric, direction = cache['monitor_metric']
    delta = cache.get('score_delta', _conf.score_delta)
    improved = False
    if direction == 'maximize':
        improved = score > cache['best_val_score'] + delta
    elif direction == 'minimize':
        improved = score < cache['best_val_score'] - delta

    if improved:
        cache['best_val_epoch'] = epoch
        cache['best_val_score'] = score
    return improved


def stop_training_(epoch, cache):
    monitor_metric, _ = cache['monitor_metric']
    if epoch - cache['best_val_epoch'] > cache.get('patience', cache['epochs']):
        return True

    if cache['monitor_metric'][1] == 'maximize':
        return cache['best_val_score'] == _conf.score_high
    elif cache['monitor_metric'][1] == 'minimize':
        return cache['best_val_score'] == _conf.score_low
    return False


def configure_profiler(cache, state):
    if not cache.get('profiler_configured') and _args.get('profile'):
        profiler_conf = _conf.profiler_conf_file
        if _os.path.exists(profiler_conf):
            _os.remove(profiler_conf)

        with open(profiler_conf, 'w') as conf:
            try:
                jsn = {'log_dir': state[_args['profiler_dir_key']]}
                conf.write(_json.dumps(jsn))
            except:
                _os.remove(profiler_conf)
                _tb.print_exc()
        cache['profiler_configured'] = True
