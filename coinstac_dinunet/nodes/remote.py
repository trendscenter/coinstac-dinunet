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

import numpy as _np
import torch as _torch

import coinstac_dinunet.config as _cs
import coinstac_dinunet.metrics as _metric
from coinstac_dinunet.vision import plot as _plot


class COINNRemote:
    def __init__(self, **kw):
        self.out = {}
        self.cache = kw['cache']
        self.input = kw['input']
        self.state = kw['state']

    def _aggregate_sites_info(self):
        """
        Average each sites gradients and pass it to all sites.
        """
        out = {}
        grads = []
        for site, site_vars in self.input.items():
            grad_file = self.state['baseDirectory'] + _os.sep + site + _os.sep + site_vars['grads_file']
            if _cs.is_format_torch: grads.append(_torch.load(grad_file))
            if _cs.is_format_numpy: grads.append(_np.load(grad_file, allow_pickle=True))
        out['avg_grads'] = _cs.avg_grads_file
        avg_grads = []

        if _cs.is_format_numpy:
            numpy_float_precision = _np.float16 if str(_cs.float_precision).endswith('float16') else _np.float
            for layer_grad in zip(*grads):
                avg_grads.append(_np.array(layer_grad, dtype=numpy_float_precision).mean(0))
            _np.save(self.state['transferDirectory'] + _os.sep + out['avg_grads'], _np.array(avg_grads))
        elif _cs.is_format_torch:
            for layer_grad in zip(*grads):
                """
                RuntimeError: "sum_cpu" not implemented for 'Half' so must convert to float32.
                """
                layer_grad = [lg.type(_torch.float32).cpu() for lg in layer_grad]
                avg_grads.append(_torch.stack(layer_grad).mean(0).type(_cs.float_precision))
            _torch.save(avg_grads, self.state['transferDirectory'] + _os.sep + out['avg_grads'])

        return out

    def _init_runs(self):
        self.cache.update(id=[v['id'] for _, v in self.input.items()][0])
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
                'id'] + _os.sep + f"fold_{self.cache['fold']['split_ix']}")
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
            for tl, tp in site_vars['train_log']:
                train_loss.add(tl['sum'], tl['count'])
                train_scores.update(tp=tp['tp'], tn=tp['tn'], fn=tp['fn'], fp=tp['fp'])
            vl, vp = site_vars['validation_log']
            val_loss.add(vl['sum'], vl['count'])
            val_scores.update(tp=vp['tp'], tn=vp['tn'], fn=vp['fn'], fp=vp['fp'])

        self.cache['train_log'].append([*train_loss.get(), *train_scores.get()])
        self.cache['validation_log'].append([*val_loss.get(), *val_scores.get()])
        self._save_if_better(val_scores)
        _plot.plot_progress(self.cache, self.cache['log_dir'], log_headers=self._get_log_headers(),
                            plot_keys=['train_log', 'validation_log'])
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
            vl, vp = site_vars['test_log']
            test_loss.add(vl['sum'], vl['count'])
            test_scores.update(tp=vp['tp'], tn=vp['tn'], fn=vp['fn'], fp=vp['fp'])
        self.cache['test_log'].append([*test_loss.get(), *test_scores.get()])
        self.cache['test_scores'] = _json.dumps(vars(test_scores))
        self.cache['global_test_score'].append(vars(test_scores))
        _plot.plot_progress(self.cache, self.cache['log_dir'], plot_keys=['train_log', 'validation_log'],
                            log_headers=self._get_log_headers())
        _plot.save_scores(self.cache, self.cache['log_dir'], file_keys=['test_log', 'test_scores'])

    def _send_global_scores(self):
        out = {}
        score = self._new_metrics()
        for sc in self.cache['global_test_score']:
            score.update(tp=sc['tp'], tn=sc['tn'], fn=sc['fn'], fp=sc['fp'])
        self.cache['global_test_score'] = ['Precision,Recall,F1,Accuracy']
        self.cache['global_test_score'].append(score.prfa())
        _plot.save_scores(self.cache, self.state['outputDirectory'] + _os.sep + self.cache['id'],
                          log_header=self._get_log_headers(), file_keys=['global_test_score'])
        _plot.save_cache(self.cache, self.state['log_dir'], self.cache['id'])

        out['results_zip'] = f"{self.cache['id']}_" + '_'.join(str(_datetime.datetime.now()).split(' '))
        _shutil.make_archive(f"{self.state['transferDirectory']}{_os.sep}{out['results_zip']}",
                            'zip', self.cache['log_dir'])
        return out

    def _set_mode(self, mode=None):
        out = {}
        for site, site_vars in self.input.items():
            out[site] = mode if mode else site_vars['mode']
        return out

    def compute(self):

        nxt_phase = self.input.get('phase', 'init_runs')
        if self._check(all, 'phase', 'init_runs', self.input):
            """
            Initialize all folds and loggers
            """
            self.cache['global_test_score'] = []
            self._init_runs()
            self.out['run'] = self._next_run()
            self.out['global_modes'] = self._set_mode()
            nxt_phase = 'init_nn'

        if self._check(all, 'phase', 'computation', self.input):
            """
            Main computation phase where we aggregate sites information
            We also handle train/validation/test stages of local sites by sending corresponding signals from here
            """
            nxt_phase = 'computation'
            self.out['global_modes'] = self._set_mode()
            if self._check(all, 'grads_file', _cs.grads_file, self.input):
                self.out.update(**self._aggregate_sites_info())

            if self._check(all, 'mode', 'val_waiting', self.input):
                self.out['global_modes'] = self._set_mode(mode='validation')

            if self._check(all, 'mode', 'train_waiting', self.input):
                self.out.update(**self._on_epoch_end())
                self.out['global_modes'] = self._set_mode(mode='train')

            if self._check(all, 'mode', 'test', self.input):
                self.out.update(**self._on_epoch_end())
                self.out['global_modes'] = self._set_mode(mode='test')

        if self._check(all, 'phase', 'next_run_waiting', self.input):
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
                nxt_phase = 'init_nn'
            else:
                self.out.update(**self._send_global_scores())
                nxt_phase = 'success'

        self.out['phase'] = nxt_phase

    def send(self):
        output = _json.dumps(
            {'output': self.out, 'cache': self.cache, 'success': self._check(all, 'phase', 'success', self.input)})
        _sys.stdout.write(output)
