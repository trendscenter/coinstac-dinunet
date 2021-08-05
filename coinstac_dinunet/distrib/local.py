"""
@author: Aashis Khanal
@email: sraashis@gmail.com
"""

import json as _json
import os as _os
import shutil as _shutil
from os import sep as _sep
from typing import List as _List

import coinstac_dinunet.config as _conf
from coinstac_dinunet.config.keys import *
from coinstac_dinunet.data import COINNDataHandle as _DataHandle
from coinstac_dinunet.distrib import learner as _learner
from coinstac_dinunet.utils import FrozenDict as _FrozenDict


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
                 pin_memory: bool = False,
                 num_workers: int = 0,
                 load_limit: int = _conf.max_size,
                 load_sparse=False,
                 pretrained_path: str = None,
                 patience: int = None,
                 num_folds: int = None,
                 split_ratio: _List[float] = None,
                 pretrain_args: dict = None,
                 dataloader_args: dict = None,
                 **kw):

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
        self._args['load_sparse'] = load_sparse
        self._args['pretrained_path'] = pretrained_path
        self._args['patience'] = patience if patience else epochs
        self._args['num_folds'] = num_folds
        self._args['split_ratio'] = split_ratio
        self._args.update(**kw)
        self._args = _FrozenDict(self._args)
        self._pretrain_args = pretrain_args if pretrain_args else {}
        self._dataloader_args = dataloader_args if dataloader_args else {}

    def _check_args(self):
        assert self.cache['computation_id'] is not None, self._PROMPT_TASK_
        assert self.cache['mode'] in [Mode.TRAIN, Mode.TEST], self._PROMPT_MODE_

    def _init_runs(self, learner):
        out = {}
        """Data related initializations."""
        out.update(learner.trainer.data_handle.prepare_data())
        out['data_size'] = {}
        for k, sp in self.cache['splits'].items():
            sp = _json.loads(open(self.cache['split_dir'] + _os.sep + sp).read())
            out['data_size'][k] = dict((key, len(sp[key])) for key in sp)
        for k in SHARED_CACHE:
            out[k] = self.cache.get(k)
        return out

    def _attach_global(self, learner):
        self.cache['nn'] = learner.trainer.nn
        self.cache['device'] = learner.trainer.device
        self.cache['optimizer'] = learner.trainer.optimizer
        self.cache['dataset'] = learner.trainer.data_handle.dataset

    def _next_run(self, learner):
        out = {}
        learner.trainer.init_nn(init_weights=True)
        self.cache['best_nn_state'] = f"best.{self.cache['computation_id']}-{self.cache['split_ix']}.pt"
        out['phase'] = Phase.COMPUTATION
        self._attach_global(learner)
        return out

    def _pretrain_local(self, trainer_cls, train_dataset, validation_dataset):
        out = {'phase': Phase.COMPUTATION}
        if self._pretrain_args.get('epochs') and self.cache['pretrain']:
            cache = {**self.cache}
            cache.update(**self._pretrain_args)
            trainer = trainer_cls(cache=cache, input=self.input, state=self.state)
            trainer.init_nn()

            trainer.init_training_cache()
            out.update(**trainer.train_local(train_dataset, validation_dataset))
            out['phase'] = Phase.PRE_COMPUTATION

        if self._pretrain_args.get('epochs') and any([r['pretrain'] for r in self.input['global_runs'].values()]):
            out['phase'] = Phase.PRE_COMPUTATION
        return out

    def compute(self, trainer_cls,
                dataset_cls=None,
                datahandle_cls=_DataHandle,
                learner_cls=_learner.COINNLearner,
                **kw):

        learner = learner_cls(
            trainer=trainer_cls(
                data_handle=datahandle_cls(
                    cache=self.cache, input=self.input, state=self.state,
                    dataloader_args=self._dataloader_args
                )
            )
        )

        self.out['phase'] = self.input.get('phase', Phase.INIT_RUNS)
        if self.out['phase'] == Phase.INIT_RUNS:
            """ Generate folds as specified.   """
            self.cache.update(**self.input)
            for k in self._args:
                if self.cache.get(k) is None:
                    self.cache[k] = self._args[k]
            self.out.update(**self._init_runs(learner))
            self.cache['args'] = _FrozenDict({**self.cache})
            self.cache['verbose'] = False
            self._check_args()

        elif self.out['phase'] == Phase.NEXT_RUN:
            self.cache.update(**self.input['global_runs'][self.state['clientId']])
            self.cache.update(cursor=0)
            self.cache[Key.TRAIN_SERIALIZABLE] = []

            self.cache['split_file'] = self.cache['splits'][self.cache['split_ix']]
            self.cache['log_dir'] = self.state['outputDirectory'] + _sep + self.cache[
                'computation_id'] + _sep + f"fold_{self.cache['split_ix']}"
            _os.makedirs(self.cache['log_dir'], exist_ok=True)

            self.out.update(**self._next_run(learner))
            self.out.update(
                **self._pretrain_local(
                    trainer_cls,
                    learner.trainer.data_handle.get_train_dataset(dataset_cls),
                    learner.trainer.data_handle.get_validation_dataset(dataset_cls))
            )

        elif self.out['phase'] == Phase.PRE_COMPUTATION and self.input.get('pretrained_weights'):
            learner.trainer.load_checkpoint(
                file_path=self.state['baseDirectory'] + _sep + self.input['pretrained_weights']
            )
            self.out['phase'] = Phase.COMPUTATION

        """Track global state among sites."""
        self.out['mode'] = learner.global_modes.get(self.state['clientId'], self.cache['mode'])

        """Computation begins..."""
        if self.out['phase'] == Phase.COMPUTATION:

            """ Train/validation and test phases """
            if self.input.get('save_current_as_best'):
                learner.trainer.save_checkpoint(
                    file_path=self.cache['log_dir'] + _sep + self.cache['best_nn_state']
                )

            """Initialize Learner and assign trainer"""
            if self.input.get('update'):
                self.out.update(**learner.step())

            if any(m == Mode.TRAIN for m in learner.global_modes.values()):
                """
                All sites must begin/resume the training the same time.
                To enforce this, we have a 'val_waiting' status. Lagged sites will go to this status, 
                   and reshuffle the data,
                take part in the training with everybody until all sites go to 'val_waiting' status.
                """
                it, out = learner.to_reduce()
                self.out.update(**out)
                if it.get('averages') and it.get('metrics'):
                    self.cache[Key.TRAIN_SERIALIZABLE].append([vars(it['averages']), vars(it['metrics'])])
                    self.out.update(**learner.trainer.on_iteration_end(0, 0, it))

            if all(m == Mode.VALIDATION for m in learner.global_modes.values()):
                """
                Once all sites are in 'val_waiting' status, remote issues 'validation' signal.
                Once all sites run validation phase, they go to 'train_waiting' status.
                Once all sites are in this status, remote issues 'train' signal
                 and all sites reshuffle the indices and r esume training.
                We send the confusion matrix to the remote to accumulate global score for model selection.
                """
                self.out.update(**learner.trainer.validation_distributed(dataset_cls))
                self.out['mode'] = Mode.TRAIN_WAITING

            if all(m == Mode.TEST for m in learner.global_modes.values()):
                self.out.update(**learner.trainer.test_distributed(dataset_cls))
                self.out['mode'] = self.cache['args']['mode']
                self.out['phase'] = Phase.NEXT_RUN_WAITING

        elif self.out['phase'] == Phase.SUCCESS:
            """ This phase receives global scores from the aggregator."""
            _shutil.copy(f"{self.state['baseDirectory']}{_sep}{self.input['results_zip']}.zip",
                         f"{self.state['outputDirectory'] + _sep + self.cache['computation_id']}{_sep}"
                         f"{self.input['results_zip']}.zip")

    def __call__(self, *args, **kwargs):
        self.compute(*args, **kwargs)
        return {'output': self.out}
