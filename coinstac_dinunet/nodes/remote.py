"""
@author: Aashis Khanal
@email: sraashis@gmail.com
@ref: https://github.com/sraashis/easytorch
"""

import datetime as _datetime
import json as _json
import math as _math
import os as _os
import shutil as _shutil
import sys as _sys
from typing import Callable as _Callable

import numpy as _np

import coinstac_dinunet.config as _conf
import coinstac_dinunet.metrics as _metric
import coinstac_dinunet.utils as _utils
import coinstac_dinunet.utils.tensorutils as _tu
from coinstac_dinunet.config.status import *
from coinstac_dinunet.vision import plotter as _plot


def average_sites_gradients(cache, input, state):
    """
    Average each sites gradients and pass it to all sites.
    """
    out = {'avg_grads_file': _conf.avg_grads_file}
    grads = []
    for site, site_vars in input.items():
        grads_file = state['baseDirectory'] + _os.sep + site + _os.sep + site_vars['grads_file']
        grads.append(_tu.load_grads(grads_file))

    avg_grads = []
    for layer_grad in zip(*grads):
        avg_grads.append(_np.array(layer_grad).mean(0))
    _tu.save_grads(state['transferDirectory'] + _os.sep + out['avg_grads_file'], avg_grads)
    return out


class COINNRemote:
    def __init__(self, cache: dict = None, input: dict = None, state: dict = None,
                 sites_reducer: _Callable = average_sites_gradients, **kw):
        self.out = {}
        self.cache = cache
        self.input = _utils.FrozenDict(input)
        self.state = _utils.FrozenDict(state)
        self.sites_reducer = sites_reducer

    def _init_runs(self):
        self.cache.update(computation_id=[v['computation_id'] for _, v in self.input.items()][0])
        self.cache.update(num_folds=[v['num_folds'] for _, v in self.input.items()][0])
        self.cache.update(seed=[v.get('seed') for _, v in self.input.items()][0])
        self.cache.update(seed=_conf.current_seed)

        self.cache['global_test_scores'] = []

        self.cache['data_size'] = {}
        for site, site_vars in self.input.items():
            self.cache['data_size'][site] = site_vars.get('data_size')

        self.cache['folds'] = []
        for fold in range(self.cache['num_folds']):
            self.cache['folds'].append({'split_ix': str(fold), 'seed': self.cache['seed']})

        self.cache['folds'] = self.cache['folds'][::-1]

    def _next_run(self):
        """
        This function pops a new fold, lock parameters, and forward init_nn signal to all sites
        """
        self.cache['fold'] = self.cache['folds'].pop()

        self._set_log_headers()
        self._set_monitor_metric()

        self.cache.update(
            log_dir=self.state['outputDirectory'] + _os.sep + self.cache[
                'computation_id'] + _os.sep + f"fold_{self.cache['fold']['split_ix']}")
        _os.makedirs(self.cache['log_dir'], exist_ok=True)

        metric_direction = self.cache['monitor_metric'][1]
        self.cache.update(best_val_score=0 if metric_direction == 'maximize' else _conf.data_load_lim)
        self.cache.update(train_scores=[], validation_scores=[], test_scores=[])

        """**** Parameter Lock ******"""
        out = {}
        for site, site_vars in self.input.items():
            """Send pretrain signal to site with maximum training data."""
            fold = {**self.cache['fold']}
            data_sizes = dict([(st, self.cache['data_size'][st][fold['split_ix']]['train']) for st in self.input])
            max_data_site = max(data_sizes, key=data_sizes.get)
            fold['pretrain'] = site == max_data_site
            out[site] = fold
        return out

    @staticmethod
    def _check(logic, k, v, kw):
        phases = []
        for site_vars in kw.values():
            phases.append(site_vars.get(k) == v)
        return logic(phases)

    def _new_metrics(self):
        return _metric.Prf1a()

    def _new_averages(self):
        return _metric.COINNAverages()

    def _set_log_headers(self):
        self.cache['log_header'] = 'Loss,Precision,Recall,F1,Accuracy'

    def _set_monitor_metric(self):
        self.cache['monitor_metric'] = 'f1', 'maximize'

    def _on_epoch_end(self):
        """
        #############################
        Entry status: "train_waiting"
        Exit status: "train"
        ############################

        This function runs once an epoch of training is done and all sites
            run the validation step i.e. all sites in "train_waiting" status.
        We accumulate training/validation loss and scores of the last epoch.
        We also send a save current model as best signal to all sites if the global validation score is better than the previously saved one.
        """
        out = {}
        val_metrics = self._new_metrics()
        val_averages = self._new_averages()
        for site, site_vars in self.input.items():
            for ta, tm in site_vars['train_scores']:
                self.cache['train_scores'].append([*ta.get(), *tm.get()])
            va, vm = site_vars['validation_scores']
            val_averages.update(**va)
            val_metrics.update(**vm)

        self.cache['validation_scores'].append([*val_averages.get(), *val_metrics.get()])
        self._save_if_better(val_metrics)

        epoch = site_vars['epoch']
        if epoch % int(_math.log(epoch + 1) + 1) == 0:
            _plot.plot_progress(self.cache, self.cache['log_dir'],
                                plot_keys=['train_scores', 'validation_scores'],
                                epoch=epoch)
        return out

    def _save_if_better(self, metrics):
        r"""
        Save the current model as best if it has better validation scores.
        """
        monitor_metric, direction = self.cache['monitor_metric']
        sc = getattr(metrics, monitor_metric)
        if callable(sc):
            sc = sc()

        if (direction == 'maximize' and sc >= self.cache['best_val_score']) or (
                direction == 'minimize' and sc <= self.cache['best_val_score']):
            self.out['save_current_as_best'] = True
            self.cache['best_val_score'] = sc
        else:
            self.out['save_current_as_best'] = False

    def _save_scores(self):
        """
        ########################
        Entry: phase "next_run_waiting"
        Exit: continue on next fold if folds left, else success and stop
        #######################
        This function saves test score of last fold.
        """
        test_averages = self._new_averages()
        test_metrics = self._new_metrics()
        for site, site_vars in self.input.items():
            ta, tm = site_vars['test_scores']
            test_averages.update(**ta)
            test_metrics.update(**tm)

        self.cache['test_scores'].append([self.cache['fold']['split_ix'], *test_averages.get(), *test_metrics.get()])
        self.cache['global_test_scores'].append([vars(test_averages), vars(test_metrics)])

        _plot.plot_progress(self.cache, self.cache['log_dir'],
                            plot_keys=['train_scores', 'validation_scores'], epoch=site_vars['epoch'])
        _utils.save_scores(self.cache, self.cache['log_dir'], file_keys=['test_scores'])
        _utils.save_cache(self.cache, self.cache['log_dir'])

    def _send_global_scores(self):
        out = {}
        averages = self._new_averages()
        metrics = self._new_metrics()
        for avg, sc in self.cache['global_test_scores']:
            averages.update(**avg)
            metrics.update(**sc)

        self.cache['_global_test_scores'] = [['Global', *averages.get(), *metrics.get()]]
        _utils.save_scores(self.cache, self.state['outputDirectory'] + _os.sep + self.cache['computation_id'],
                           file_keys=['_global_test_scores'])

        out['results_zip'] = f"{self.cache['computation_id']}_" + '_'.join(str(_datetime.datetime.now()).split(' '))
        _shutil.make_archive(f"{self.state['transferDirectory']}{_os.sep}{out['results_zip']}",
                             'zip', self.state['outputDirectory'] + _os.sep + self.cache['computation_id'])
        return out

    def _set_mode(self, mode=None):
        out = {}
        for site, site_vars in self.input.items():
            out[site] = mode if mode else site_vars.get('mode', 'N/A')
        return out

    def _pre_compute(self):
        out = {}
        pt_path = None
        for site, site_vars in self.input.items():
            if site_vars.get('weights_file') is not None:
                pt_path = self.state['baseDirectory'] + _os.sep + site + _os.sep + site_vars['weights_file']
                break
        if pt_path is not None:
            out['pretrained_weights'] = f'pretrained_{_conf.weights_file}'
            _shutil.copy(pt_path, self.state['transferDirectory'] + _os.sep + out['pretrained_weights'])
        return out

    def compute(self):

        self.out['phase'] = self.input.get('phase', Phase.INIT_RUNS)
        if self._check(all, 'phase', Phase.INIT_RUNS, self.input):
            """
            Initialize all folds and loggers
            """
            self._init_runs()
            self.out['global_runs'] = self._next_run()
            self.out['phase'] = Phase.INIT_NN

        if self._check(all, 'phase', Phase.PRE_COMPUTATION, self.input):
            self.out.update(**self._pre_compute())
            self.out['phase'] = Phase.PRE_COMPUTATION

        self.out['global_modes'] = self._set_mode()
        if self._check(all, 'phase', Phase.COMPUTATION, self.input):
            """
            Main computation phase where we aggregate sites information
            We also handle train/validation/test stages of local sites by sending corresponding signals from here
            """
            self.out['phase'] = Phase.COMPUTATION
            if self._check(all, 'grads_file', _conf.grads_file, self.input):
                self.out.update(**self.sites_reducer(self.cache, self.input, self.state))

            if self._check(all, 'mode', Mode.VALIDATION_WAITING, self.input):
                self.out['global_modes'] = self._set_mode(mode=Mode.VALIDATION)

            if self._check(all, 'mode', Mode.TRAIN_WAITING, self.input):
                self.out.update(**self._on_epoch_end())
                self.out['global_modes'] = self._set_mode(mode=Mode.TRAIN)

            if self._check(all, 'mode', Mode.TEST, self.input):
                self.out.update(**self._on_epoch_end())
                self.out['global_modes'] = self._set_mode(mode=Mode.TEST)

        if self._check(all, 'phase', Phase.NEXT_RUN_WAITING, self.input):
            """
            This block runs when a fold has completed all train, test, validation phase.
            We save all the scores and plot the results.
            We transition to new fold if there is any left, else we stop the distributed computation with a success signal.
            """
            self._save_scores()
            if len(self.cache['folds']) > 0:
                self.out['nn'] = {}
                self.out['global_runs'] = self._next_run()
                self.out['phase'] = Phase.INIT_NN
            else:
                self.out.update(**self._send_global_scores())
                self.out['phase'] = Phase.SUCCESS

    def send(self):
        output = _json.dumps(
            {'output': self.out, 'cache': self.cache,
             'success': self._check(all, 'phase', Phase.SUCCESS, self.input)})
        _sys.stdout.write(output)
