from coinstac_dinunet import config as _conf


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


def stop_training_(epoch, score, cache):
    monitor_metric, _ = cache['monitor_metric']
    if epoch - cache['best_val_epoch'] > cache.get('patience', cache['epochs']):
        return True
    if epoch % cache.get('score_window_len', _conf.score_window_len) == 0:
        avg = sum(cache['score_window']) / len(cache['score_window'])
        cache['score_window'] = []
        return abs(avg - cache['best_val_score']) <= cache.get('score_delta', _conf.score_delta)
    else:
        cache['score_window'].append(score)
    return False
