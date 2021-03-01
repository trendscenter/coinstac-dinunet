"""
@author: Aashis Khanal
@email: sraashis@gmail.com
@ref: https://github.com/sraashis/easytorch
"""

import json as _json
import os as _os
import shutil as _shutil
import sys as _sys
from os import sep as _sep
from typing import List as _List, Callable as _Callable

import coinstac_dinunet.config as _conf
import coinstac_dinunet.data.datautils as _du
from coinstac_dinunet.config.status import *
from coinstac_dinunet.utils import FrozenDict as _FrozenDict
import random as _rd


class COINNLocal:
    _PROMPT_TASK_ = "Task name must be given."
    _PROMPT_MODE_ = f"Mode must be provided and should be one of {[Mode.TRAIN, Mode.TEST]}."

    def __init__(self, cache: dict = None, input: dict = None, state: dict = None,
                 computation_id=None,
                 mode: str = None,
                 batch_size: int = 16,
                 local_iterations: int = 1,
                 epochs: int = 31,
                 validation_epochs: int = 1,
                 learning_rate: float = 0.001,
                 gpus: _List[int] = None,
                 pin_memory: bool = _conf.gpu_available,
                 num_workers: int = 0,
                 load_limit: int = _conf.max_size,
                 pretrained_path: str = None,
                 patience: int = None,
                 num_folds: int = None,
                 split_ratio: _List[float] = None,
                 pretrain_args: dict = None, **kw):
        self.out = {}
        self.cache = cache
        self.input = _FrozenDict(input)
        self.state = _FrozenDict(state)
        self._args = {}
        self._args['computation_id'] = computation_id  #
        self._args['mode'] = mode  # test/train
        self._args['batch_size'] = batch_size
        self._args['local_iterations'] = local_iterations
        self._args['epochs'] = epochs
        self._args['validation_epochs'] = validation_epochs
        self._args['learning_rate'] = learning_rate
        self._args['gpus'] = gpus
        self._args['pin_memory'] = pin_memory
        self._args['num_workers'] = num_workers
        self._args['load_limit'] = load_limit
        self._args['pretrained_path'] = pretrained_path
        self._args['patience'] = patience if patience else epochs
        self._args['num_folds'] = num_folds
        self._args['split_ratio'] = split_ratio
        self._args.update(**kw)
        self._args = _FrozenDict(self._args)
        self._pretrain_args = pretrain_args if pretrain_args else {}
        self._GLOBAL_STATE = {}

    def _check_args(self):
        assert self.cache['computation_id'] is not None, self._PROMPT_TASK_
        assert self.cache['mode'] in [Mode.TRAIN, Mode.TEST], self._PROMPT_MODE_

    def _init_runs(self):
        out = {}
        out.update(_du.init_k_folds(self.cache, self.state))
        out['data_size'] = {}
        for k, sp in self.cache['splits'].items():
            sp = _json.loads(open(self.cache['split_dir'] + _os.sep + sp).read())
            out['data_size'][k] = dict((key, len(sp[key])) for key in sp)
        for k in SHARED_CACHE:
            out[k] = self.cache.get(k)
        return out

    def _init_nn_state(self, trainer):
        out = {}
        self.cache['current_nn_state'] = 'current.nn.pt'
        self.cache['best_nn_state'] = 'best.nn.pt'
        trainer.init_nn(init_weights=True)
        trainer.save_checkpoint(file_path=self.cache['log_dir'] + _sep + self.cache['current_nn_state'])
        out['phase'] = Phase.COMPUTATION
        return out

    def _pretrain_local(self, trainer_cls, dataset_cls):
        out = {}
        if self._pretrain_args.get('epochs') and self.cache['pretrain']:
            cache = {**self.cache}
            cache.update(**self._pretrain_args)
            trainer = trainer_cls(cache=cache, input=self.input, state=self.state)
            trainer.init_nn()
            trainer.init_training_cache()
            out.update(**trainer.train_local(dataset_cls))
            _rd.shuffle(trainer.cache.get('data_indices', []))
            out['phase'] = Phase.PRE_COMPUTATION

        if self._pretrain_args.get('epochs') and any([r['pretrain'] for r in self._GLOBAL_STATE['runs'].values()]):
            out['phase'] = Phase.PRE_COMPUTATION
        return out

    def compute(self, dataset_cls, trainer_cls):
        self.out['phase'] = self.input.get('phase', Phase.INIT_RUNS)
        trainer = trainer_cls(cache=self.cache, input=self.input, state=self.state)

        if self.out['phase'] == Phase.INIT_RUNS:
            """ Generate folds as specified.   """
            self.cache.update(**self.input)
            for k in self._args:
                if self.cache.get(k) is None:
                    self.cache[k] = self._args[k]
            self.out.update(**self._init_runs())
            self.cache['args'] = _FrozenDict({**self.cache})
            self.cache['verbose'] = False
            self._check_args()

        elif self.out['phase'] == Phase.INIT_NN:
            """  Initialize neural network/optimizer and GPUs  """
            self._GLOBAL_STATE['runs'] = self.input['global_runs']
            self.cache.update(**self._GLOBAL_STATE['runs'][self.state['clientId']])
            self.cache.update(cursor=0)
            self.cache[Key.TRAIN_SERIALIZABLE] = []

            self.cache['split_file'] = self.cache['splits'][self.cache['split_ix']]
            self.cache['log_dir'] = self.state['outputDirectory'] + _sep + self.cache[
                'computation_id'] + _sep + f"fold_{self.cache['split_ix']}"
            _os.makedirs(self.cache['log_dir'], exist_ok=True)
            self.out.update(**self._init_nn_state(trainer))
            trainer.cache_data_indices(dataset_cls, split_key='train')
            self.out.update(**self._pretrain_local(trainer_cls, dataset_cls))

        elif self.out['phase'] == Phase.PRE_COMPUTATION and self.input.get('pretrained_weights'):
            trainer.init_nn(init_weights=False)
            trainer.load_checkpoint(file_path=self.state['baseDirectory'] + _sep + self.input['pretrained_weights'])
            trainer.save_checkpoint(file_path=self.cache['log_dir'] + _sep + self.cache['current_nn_state'])
            self.out['phase'] = Phase.COMPUTATION

        """################################### Computation ##########################################"""
        self._GLOBAL_STATE['modes'] = self.input.get('global_modes', {})
        self.out['mode'] = self._GLOBAL_STATE['modes'].get(self.state['clientId'], self.cache['mode'])

        if self.out['phase'] == Phase.COMPUTATION:
            """ Train/validation and test phases """
            trainer.init_nn(init_weights=False)
            trainer.load_checkpoint(file_path=self.cache['log_dir'] + _sep + self.cache['current_nn_state'])

            if self.input.get('save_current_as_best'):
                trainer.save_checkpoint(file_path=self.cache['log_dir'] + _sep + self.cache['best_nn_state'])

            if self.input.get('avg_grads_file'):
                trainer.step()
                trainer.save_checkpoint(file_path=self.cache['log_dir'] + _sep + self.cache['current_nn_state'])

            if any(m == Mode.TRAIN for m in self._GLOBAL_STATE['modes'].values()):
                """
                All sites must begin/resume the training the same time.
                To enforce this, we have a 'val_waiting' status. Lagged sites will go to this status, 
                   and reshuffle the data,
                take part in the training with everybody until all sites go to 'val_waiting' status.
                """
                self.out.update(**trainer.train_distributed(dataset_cls))

            if all(m == Mode.VALIDATION for m in self._GLOBAL_STATE['modes'].values()):
                """
                Once all sites are in 'val_waiting' status, remote issues 'validation' signal.
                Once all sites run validation phase, they go to 'train_waiting' status.
                Once all sites are in this status, remote issues 'train' signal
                 and all sites reshuffle the indices and r esume training.
                We send the confusion matrix to the remote to accumulate global score for model selection.
                """
                self.out.update(**trainer.validation_distributed(dataset_cls))
                self.out['mode'] = Mode.TRAIN_WAITING

            elif all(m == Mode.TEST for m in self._GLOBAL_STATE['modes'].values()):
                self.out.update(**trainer.test_distributed(dataset_cls))
                self.out['mode'] = self.cache['args']['mode']
                self.out['phase'] = Phase.NEXT_RUN_WAITING

        elif self.out['phase'] == Phase.SUCCESS:
            """ This phase receives global scores from the aggregator."""
            _shutil.copy(f"{self.state['baseDirectory']}{_sep}{self.input['results_zip']}.zip",
                         f"{self.state['outputDirectory'] + _sep + self.cache['computation_id']}{_sep}{self.input['results_zip']}.zip")

    def send(self):
        output = _json.dumps({'output': self.out, 'cache': self.cache})
        _sys.stdout.write(output)
