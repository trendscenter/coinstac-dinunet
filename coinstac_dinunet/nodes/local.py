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


class COINNLocal:
    _PROMPT_TASK_ = "Task name must be given."
    _PROMPT_MODE_ = f"Mode must be provided and should be one of {[Mode.TRAIN, Mode.TEST]}."

    def __init__(self, cache: dict = None, input: dict = None, state: dict = None,
                 computation_id=None,
                 mode: str = None,
                 batch_size: int = 4,
                 local_iteration: int = 1,
                 pretrain_epochs: int = 10,
                 epochs: int = 21,
                 learning_rate: float = 0.001,
                 gpus: _List[int] = None,
                 pin_memory: bool = _conf.gpu_available,
                 num_workers: int = 0,
                 load_limit: int = _conf.data_load_lim,
                 pretrained_path: str = None,
                 patience: int = 5,
                 load_sparse: bool = False,
                 num_folds: int = None,
                 split_ratio: _List[float] = (0.6, 0.2, 0.2),
                 data_splitter: _Callable = None, **kw):
        self.out = {}
        self.cache = cache
        self.input = _FrozenDict(input)
        self.state = _FrozenDict(state)
        self.data_splitter = data_splitter
        self.args = {}
        self.args['computation_id'] = computation_id  #
        self.args['mode'] = mode  # test/train
        self.args['batch_size'] = batch_size
        self.args['local_iteration'] = local_iteration
        self.args['epochs'] = epochs
        self.args['pretrain_epochs'] = pretrain_epochs
        self.args['learning_rate'] = learning_rate
        self.args['gpus'] = gpus
        self.args['pin_memory'] = pin_memory
        self.args['num_workers'] = num_workers
        self.args['load_limit'] = load_limit
        self.args['pretrained_path'] = pretrained_path
        self.args['patience'] = patience
        self.args['load_sparse'] = load_sparse
        self.args['num_folds'] = num_folds
        self.args['split_ratio'] = split_ratio
        self.args.update(**kw)
        self.args = _FrozenDict(self.args)

    def _check_args(self):
        assert self.cache['computation_id'] is not None, self._PROMPT_TASK_
        assert self.cache['mode'] in [Mode.TRAIN, Mode.TEST], self._PROMPT_MODE_

    def _init_runs(self):
        out = {}
        out.update(_du.init_k_folds(self.cache, self.state, self.data_splitter))
        out['data_size'] = {}
        for k, sp in self.cache['splits'].items():
            sp = _json.loads(open(self.cache['split_dir'] + _os.sep + sp).read())
            out['data_size'][k] = dict((key, len(sp[key])) for key in sp)
        out['computation_id'] = self.cache['computation_id']
        out['seed'] = self.cache.get('seed')
        return out

    def _init_nn(self, trainer, dataset_cls):
        trainer.init_nn(init_weights=True)
        self.cache['current_nn_state'] = 'current.nn.pt'
        self.cache['best_nn_state'] = 'best.nn.pt'
        trainer.load_data_indices(dataset_cls, split_key='train')
        trainer.save_checkpoint(file_name=self.cache['log_dir'] + _sep + self.cache['current_nn_state'])

        if self.cache['pretrain_epochs'] >= 1 and self.cache.get('pretrain', False):
            return Phase.PRE_COMPUTATION, trainer.pre_train()
        return Phase.COMPUTATION, {}

    def compute(self, dataset_cls, trainer_cls):
        trainer = trainer_cls(cache=self.cache, input=self.input, state=self.state)
        nxt_phase = self.input.get('phase', Phase.INIT_RUNS)
        if nxt_phase == Phase.INIT_RUNS:
            """ Generate folds as specified.   """
            self.cache.update(**self.input)
            for k in self.args:
                if self.cache.get(k) is None:
                    self.cache[k] = self.args[k]
            self.out.update(**self._init_runs())
            self.cache['args'] = _FrozenDict(self.cache)
            self.out['mode'] = self.cache['mode']
            self._check_args()

        elif nxt_phase == Phase.INIT_NN:
            """  Initialize neural network/optimizer and GPUs  """
            self.cache.update(**self.input['run'][self.state['clientId']], epoch=0, cursor=0, train_log=[])
            self.cache['split_file'] = self.cache['splits'][str(self.cache['split_ix'])]
            self.cache['log_dir'] = self.state['outputDirectory'] + _sep + self.cache[
                'computation_id'] + _sep + f"fold_{self.cache['split_ix']}"
            _os.makedirs(self.cache['log_dir'], exist_ok=True)
            nxt_phase, out = self._init_nn(trainer, dataset_cls)
            self.out.update(**out)

        elif nxt_phase == Phase.PRE_COMPUTATION:
            trainer.load_checkpoint(file_path=self.state['baseDirectory'] + _sep + self.input['pretrained_weights'])
            trainer.save_checkpoint(file_name=self.cache['log_dir'] + _sep + self.cache['current_nn_state'])
            nxt_phase = Phase.COMPUTATION

        if nxt_phase == Phase.COMPUTATION:
            """ Train/validation and test phases """
            self.out.update(mode=self.input['global_modes'].get(self.state['clientId'], self.cache['mode']))

            trainer.init_nn(init_weights=False)
            trainer.load_checkpoint(file_path=self.cache['log_dir'] + _sep + self.cache['current_nn_state'])

            if self.input.get('save_current_as_best'):
                trainer.save_checkpoint(file_name=self.cache['best_nn_state'])
                self.cache['best_epoch'] = self.cache['epoch']

            if any(m == Mode.TRAIN for m in self.input['global_modes'].values()):
                """
                All sites must begin/resume the training the same time.
                To enforce this, we have a 'val_waiting' status. Lagged sites will go to this status, 
                   and reshuffle the data,
                take part in the training with everybody until all sites go to 'val_waiting' status.
                """
                self.out.update(**trainer.train(dataset_cls))

            elif self.out['mode'] == Mode.VALIDATION:
                """
                Once all sites are in 'val_waiting' status, remote issues 'validation' signal.
                Once all sites run validation phase, they go to 'train_waiting' status.
                Once all sites are in this status, remote issues 'train' signal
                 and all sites reshuffle the indices and resume training.
                We send the confusion matrix to the remote to accumulate global score for model selection.
                """
                self.out.update(**trainer.validation(dataset_cls))

            elif self.out['mode'] == Mode.TEST:
                self.out.update(**trainer.test(dataset_cls))
                self.out['mode'] = self.cache['args']['mode']
                nxt_phase = Phase.NEXT_RUN_WAITING

        elif nxt_phase == Phase.SUCCESS:
            """ This phase receives global scores from the aggregator."""
            _shutil.copy(f"{self.state['baseDirectory']}{_sep}{self.input['results_zip']}.zip",
                         f"{self.state['outputDirectory'] + _sep + self.cache['computation_id']}{_sep}{self.input['results_zip']}.zip")

        self.out['phase'] = nxt_phase

    def send(self):
        output = _json.dumps({'output': self.out, 'cache': self.cache})
        _sys.stdout.write(output)
