"""
@author: Aashis Khanal
@email: sraashis@gmail.com
"""

import datetime as _datetime
import os as _os
import shutil as _shutil

import coinstac_dinunet.config as _conf
import coinstac_dinunet.utils as _utils
from coinstac_dinunet.config.keys import *
from coinstac_dinunet.utils import performance_improved_, stop_training_
from coinstac_dinunet.utils.logger import *
from coinstac_dinunet.vision import plotter as _plot
from .reducer import COINNReducer as _dSGDReducer
from .utils import DADReducer as _DADReducer, check as _check


class EmptyDataHandle:
    def __init__(self, cache, input, state):
        self.cache = cache
        self.input = input
        self.state = state


class COINNRemote:
    def __init__(self, cache: dict = None, input: dict = None, state: dict = None, verbose=False, **kw):
        self.out = {}

        self.cache = cache
        cache.update(**kw)

        self.input = _utils.FrozenDict(input)
        self.state = _utils.FrozenDict(state)

        self.cache['verbose'] = verbose
        if not self.cache.get(Key.ARGS_CACHED):
            site = list(self.input.values())[0]
            self.cache.update(**site['shared_args'])
            self.cache[Key.ARGS_CACHED] = True

    def _init_runs(self):
        self.cache.update(seed=_conf.current_seed)
        self.cache[Key.GLOBAL_TEST_SERIALIZABLE] = []

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
        self.cache.update(
            log_dir=self.state['outputDirectory'] + _os.sep + self.cache[
                'task_id'] + _os.sep + f"fold_{self.cache['fold']['split_ix']}")
        _os.makedirs(self.cache['log_dir'], exist_ok=True)

        metric_direction = self.cache['metric_direction']
        self.cache.update(epoch=0, best_val_epoch=0)
        self.cache.update(best_val_score=0 if metric_direction == 'maximize' else _conf.max_size)
        self.cache[Key.TRAIN_LOG] = []
        self.cache[Key.VALIDATION_LOG] = []
        self.cache[Key.TEST_METRICS] = []

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

    def _accumulate_epoch_info(self, reducer):
        out = {}
        val_averages, val_metrics = reducer.trainer.new_averages(), reducer.trainer.new_metrics()
        train_averages, train_metrics = reducer.trainer.new_averages(), reducer.trainer.new_metrics()
        for site, site_vars in self.input.items():
            for ta, tm in site_vars[Key.TRAIN_SERIALIZABLE]:
                train_averages.update(**ta), train_metrics.update(**tm)

            if site_vars.get(Key.VALIDATION_SERIALIZABLE):
                va, vm = site_vars[Key.VALIDATION_SERIALIZABLE]
                val_averages.update(**va), val_metrics.update(**vm)
        out['train_averages'] = train_averages
        out['train_metrics'] = train_metrics
        if site_vars.get(Key.VALIDATION_SERIALIZABLE):
            out['val_averages'] = val_averages
            out['val_metrics'] = val_metrics
        return out

    def _on_run_end(self, trainer):
        """
        ########################
        Entry: phase "next_run_waiting"
        Exit: continue on next fold if folds left, else success and stop
        #######################
        This function saves test score of last fold.
        """
        test_averages, test_metrics = trainer.new_averages(), trainer.new_metrics()
        for site, site_vars in self.input.items():
            if site_vars.get(Key.TEST_SERIALIZABLE):
                ta, tm = site_vars[Key.TEST_SERIALIZABLE]
                test_averages.update(**ta), test_metrics.update(**tm)

        if site_vars.get(Key.TEST_SERIALIZABLE):
            self.cache[Key.TEST_METRICS].append([*test_averages.get(), *test_metrics.get()])
            self.cache[Key.GLOBAL_TEST_SERIALIZABLE].append([vars(test_averages), vars(test_metrics)])

        _plot.plot_progress(self.cache, self.cache['log_dir'], plot_keys=[Key.TRAIN_LOG, Key.VALIDATION_LOG])
        _utils.save_scores(self.cache, self.cache['log_dir'], file_keys=[Key.TEST_METRICS])

        _cache = {**self.cache}
        _cache[Key.GLOBAL_TEST_SERIALIZABLE] = _cache[Key.GLOBAL_TEST_SERIALIZABLE][-1]
        _utils.save_cache(_cache, self.cache['log_dir'])

    def _send_global_scores(self, trainer):
        out = {}
        averages = trainer.new_averages()
        metrics = trainer.new_metrics()
        for avg, sc in self.cache[Key.GLOBAL_TEST_SERIALIZABLE]:
            averages.update(**avg)
            metrics.update(**sc)

        self.cache[Key.GLOBAL_TEST_METRICS] = [[*averages.get(), *metrics.get()]]
        _utils.save_scores(self.cache, self.state['outputDirectory'] + _os.sep + self.cache['task_id'],
                           file_keys=[Key.GLOBAL_TEST_METRICS])

        out['results_zip'] = f"{self.cache['task_id']}_" + '_'.join(str(_datetime.datetime.now()).split(' '))
        _shutil.make_archive(f"{self.state['transferDirectory']}{_os.sep}{out['results_zip']}",
                             'zip', self.state['outputDirectory'] + _os.sep + self.cache['task_id'])
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

    def compute(self, trainer_cls=None, reducer_cls: callable = _dSGDReducer, **kw):
        trainer = trainer_cls(
            data_handle=EmptyDataHandle(cache=self.cache, input=self.input, state=self.state)
        )
        self.out['phase'] = self.input.get('phase', Phase.INIT_RUNS)
        if _check(all, 'phase', Phase.INIT_RUNS, self.input):
            """
            Initialize all folds and loggers
            """
            self._init_runs()
            self.out['global_runs'] = self._next_run()
            self.out['phase'] = Phase.NEXT_RUN

        if _check(all, 'phase', Phase.PRE_COMPUTATION, self.input):
            self.out.update(**self._pre_compute())
            self.out['phase'] = Phase.PRE_COMPUTATION

        self.out['global_modes'] = self._set_mode()
        if _check(all, 'phase', Phase.COMPUTATION, self.input):
            """Initialize reducer"""
            reducer = self._get_reducer_cls(reducer_cls)(trainer=trainer)

            self.out['phase'] = Phase.COMPUTATION
            if _check(all, 'reduce', True, self.input):
                self.out.update(**reducer.reduce())

            if _check(all, 'mode', Mode.VALIDATION_WAITING, self.input):
                self.cache['epoch'] += 1
                if self.cache['epoch'] % self.cache['validation_epochs'] == 0:
                    self.out['global_modes'] = self._set_mode(mode=Mode.VALIDATION)
                else:
                    self.out['global_modes'] = self._set_mode(mode=Mode.TRAIN)

            if _check(all, 'mode', Mode.TRAIN_WAITING, self.input):
                epoch_info = self._on_epoch_end(reducer)
                nxt_epoch = self._next_epoch(**epoch_info)
                self.out['global_modes'] = self._set_mode(mode=nxt_epoch['mode'])

        if _check(all, 'phase', Phase.NEXT_RUN_WAITING, self.input):
            """
            This block runs when a fold has completed all train, test, validation phase.
            We save all the scores and plot the results.
            We transition to new fold if there is any left, else we stop with a success signal.
            """
            self._on_run_end(trainer)
            if len(self.cache['folds']) > 0:
                self.out['global_runs'] = self._next_run()
                self.out['phase'] = Phase.NEXT_RUN
            else:
                self.out.update(**self._send_global_scores(trainer))
                self.out['phase'] = Phase.SUCCESS

    def _next_epoch(self, **kw):
        out = {}
        epochs_done = self.cache['epoch'] > self.cache['epochs']
        if epochs_done or self._stop_early(**kw):
            out['mode'] = Mode.TEST
        else:
            out['mode'] = Mode.TRAIN
        return out

    def _on_epoch_end(self, reducer):
        epoch_info = self._accumulate_epoch_info(reducer)
        self.cache[Key.TRAIN_LOG].append([*epoch_info['train_averages'].get(), *epoch_info['train_metrics'].get()])
        self._save_if_better(**epoch_info)
        if epoch_info.get('val_averages'):
            self.cache[Key.VALIDATION_LOG].append([*epoch_info['val_averages'].get(), *epoch_info['val_metrics'].get()])

        if lazy_debug(self.cache['epoch']):
            _plot.plot_progress(self.cache, self.cache['log_dir'], plot_keys=[Key.TRAIN_LOG, Key.VALIDATION_LOG])
        return epoch_info

    def _save_if_better(self, **kw):
        if kw.get('val_metrics'):
            val_score = kw['val_metrics'].extract(self.cache['monitor_metric'])
            self.out['save_current_as_best'] = performance_improved_(self.cache['epoch'], val_score, self.cache)

    def _stop_early(self, **kw):
        return stop_training_(self.cache['epoch'], self.cache)

    def _get_reducer_cls(self, reducer_cls):
        if self.cache.get('agg_engine') == AGG_Engine.dSGD:
            return _dSGDReducer
        elif self.cache.get('agg_engine') == AGG_Engine.rankDAD:
            return _DADReducer

        return reducer_cls

    def __call__(self, *args, **kwargs):
        self.compute(*args, **kwargs)
        return {'output': self.out, 'success': _check(all, 'phase', Phase.SUCCESS, self.input)}
