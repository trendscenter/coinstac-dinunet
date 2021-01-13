"""
@author: Aashis Khanal
@email: sraashis@gmail.com
@ref: https://github.com/sraashis/easytorch
"""

import datetime as _datetime
import json as _json
import os as _os
import random as _random
import shutil as _shutil
import sys as _sys
from typing import Callable as _Callable

import numpy as _np
import torch as _torch

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
        if _conf.grads_numpy:
            avg_grads.append(_tu.caste_tensor(_np.array(layer_grad).mean(0)))
        if _conf.grads_torch:
            """ RuntimeError: "sum_cpu" not implemented for 'Half' so must convert to float32. """
            layer_grad = [lg.type(_torch.float32).cpu() for lg in layer_grad]
            avg_grads.append(_tu.caste_tensor(_torch.stack(layer_grad).mean(0)))
    _torch.save(avg_grads, state['transferDirectory'] + _os.sep + out['avg_grads_file'])
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
        self.cache.update(id=[v['task_name'] for _, v in self.input.items()][0])
        self.cache.update(num_folds=[v['num_folds'] for _, v in self.input.items()][0])
        self.cache.update(seed=[v.get('seed') for _, v in self.input.items()][0])
        self.cache.update(seed=_random.randint(0, int(1e6)) if self.cache['seed'] is None else self.cache['seed'])
        self.cache['folds'] = []
        for fold in range(self.cache['num_folds']):
            self.cache['folds'].append({'split_ix': fold, 'seed': self.cache['seed']})

        self.cache['folds'] = self.cache['folds'][::-1]

    def _next_run(self):
        """
        This function pops a new fold, lock parameters, and forward init_nn signal to all sites
        """
        self.cache['fold'] = self.cache['folds'].pop()
        self.cache.update(
            log_dir=self.state['outputDirectory'] + _os.sep + self.cache[
                'task_name'] + _os.sep + f"fold_{self.cache['fold']['split_ix']}")
        _os.makedirs(self.cache['log_dir'], exist_ok=True)

        metric_direction = self._monitor_metric()[1]
        self.cache.update(best_val_score=0 if metric_direction == 'maximize' else 1e11)
        self.cache.update(train_log=[], validation_log=[], test_log=[])

        """**** Parameter Lock ******"""
        out = {}
        for site, site_vars in self.input.items():
            out[site] = self.cache['fold']
        return out

    def _get_log_headers(self):
        return 'Loss,Precision,Recall,F1,Accuracy'

    @staticmethod
    def _check(logic, k, v, kw):
        phases = []
        for site_vars in kw.values():
            phases.append(site_vars.get(k) == v)
        return logic(phases)

    def _new_metrics(self):
        return _metric.Prf1a()

    def _new_averages(self):
        return _metric.ETAverages()

    def _monitor_metric(self):
        return 'f1', 'maximize'

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
        train_scores = self._new_metrics()
        train_loss = self._new_averages()
        val_scores = self._new_metrics()
        val_loss = self._new_averages()
        for site, site_vars in self.input.items():
            for ta, tm in site_vars['train_log']:
                train_loss.update(**ta)
                train_scores.update(**tm)
            va, vm = site_vars['validation_log']
            val_loss.update(**va)
            val_scores.update(**vm)

        self.cache['train_log'].append([*train_loss.get(), *train_scores.get()])
        self.cache['validation_log'].append([*val_loss.get(), *val_scores.get()])
        self._save_if_better(val_scores)
        _plot.plot_progress(self.cache, self.cache['log_dir'], plot_keys=['train_log', 'validation_log'])
        return out

    def _save_if_better(self, metrics):
        r"""
        Save the current model as best if it has better validation scores.
        """
        monitor_metric, direction = self._monitor_metric()
        sc = getattr(metrics, monitor_metric)
        if callable(sc):
            sc = sc()

        if (direction == 'maximize' and sc >= self.cache['best_score']) or (
                direction == 'minimize' and sc <= self.cache['best_score']):
            self.out['save_current_as_best'] = True
            self.cache['best_val_score'] = sc
        else:
            self.out['save_current_as_best'] = False

    def _save_test_scores(self):
        """
        ########################
        Entry: phase "next_run_waiting"
        Exit: continue on next fold if folds left, else success and stop
        #######################
        This function saves test score of last fold.
        """
        test_scores = self._new_metrics()
        test_loss = self._new_averages()
        for site, site_vars in self.input.items():
            va, vm = site_vars['test_log']
            test_loss.update(**va)
            test_scores.update(**vm)
        self.cache['test_log'].append([*test_loss.get(), *test_scores.get()])
        self.cache['test_scores'] = _json.dumps(vars(test_scores))
        self.cache['global_test_score'].append(vars(test_scores))
        _plot.plot_progress(self.cache, self.cache['log_dir'], plot_keys=['train_log', 'validation_log'])
        _utils.save_scores(self.cache, self.cache['log_dir'], file_keys=['test_log', 'test_scores'])
        _utils.save_cache(self.cache, self.state['log_dir'])

    def _send_global_scores(self):
        out = {}
        score = self._new_metrics()
        for sc in self.cache['global_test_score']:
            score.update(**sc)
        self.cache['global_test_score'] = ['Precision,Recall,F1,Accuracy']
        self.cache['global_test_score'].append(score.get())
        _utils.save_scores(self.cache, self.state['outputDirectory'] + _os.sep + self.cache['task_name'],
                           file_keys=['global_test_score'])

        out['results_zip'] = f"{self.cache['task_name']}_" + '_'.join(str(_datetime.datetime.now()).split(' '))
        _shutil.make_archive(f"{self.state['transferDirectory']}{_os.sep}{out['results_zip']}",
                             'zip', self.cache['log_dir'])
        return out

    def _set_mode(self, mode=None):
        out = {}
        for site, site_vars in self.input.items():
            out[site] = mode if mode else site_vars['mode']
        return out

    def compute(self):

        nxt_phase = self.input.get('phase', Phase.INIT_RUNS)
        if self._check(all, 'phase', Phase.INIT_RUNS, self.input):
            """
            Initialize all folds and loggers
            """
            self.cache['global_test_score'] = []
            self._init_runs()
            self.out['run'] = self._next_run()
            self.out['global_modes'] = self._set_mode()
            nxt_phase = Phase.INIT_NN

        if self._check(all, 'phase', Phase.COMPUTATION, self.input):
            """
            Main computation phase where we aggregate sites information
            We also handle train/validation/test stages of local sites by sending corresponding signals from here
            """
            nxt_phase = Phase.COMPUTATION
            self.out['global_modes'] = self._set_mode()
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

        if self._check(all, 'mode', Phase.NEXT_RUN_WAITING, self.input):
            """
            This block runs when a fold has completed all train, test, validation phase.
            We save all the scores and plot the results.
            We transition to new fold if there is any left, else we stop the distributed computation with a success signal.
            """
            self._save_test_scores()
            if len(self.cache['folds']) > 0:
                self.out['nn'] = {}
                self.out['run'] = self._next_run()
                self.out['global_modes'] = self._set_mode()
                nxt_phase = Phase.INIT_NN
            else:
                self.out.update(**self._send_global_scores())
                nxt_phase = Phase.SUCCESS

        self.out['phase'] = nxt_phase

    def send(self):
        output = _json.dumps(
            {'output': self.out, 'cache': self.cache,
             'success': self._check(all, 'phase', Phase.SUCCESS, self.input)})
        _sys.stdout.write(output)
