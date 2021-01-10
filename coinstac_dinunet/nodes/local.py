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
from typing import Callable as _Callable

import coinstac_dinunet.data.datautils as _du
from coinstac_dinunet.utils import FrozenDict as _FrozenDict


class COINNLocal:

    def __init__(self, data_splitter: _Callable = None, **kw):
        self.out = {}
        self.cache = kw['cache']
        self.input = kw['input']
        self.state = kw['state']
        self.data_splitter = data_splitter

    def compute(self, dataset_cls, trainer_cls):
        trainer = trainer_cls(cache=self.cache, input=self.input, state=self.state)
        nxt_phase = self.input.get('phase', 'init_runs')
        if nxt_phase == 'init_runs':
            """ Generate folds as specified.   """
            self.cache.update(**self.input)
            self.out.update(_du.init_k_folds(self.cache, self.state, self.data_splitter))
            self.cache['args'] = _FrozenDict(self.input)
            self.out['mode'] = self.cache['mode']

        if nxt_phase == 'init_nn':
            """  Initialize neural network/optimizer and GPUs  """

            self.cache.update(**self.input['run'][self.state['clientId']], epoch=0, cursor=0, train_log=[])
            self.cache['split_file'] = self.cache['splits'][str(self.cache['split_ix'])]
            self.cache['log_dir'] = self.state['outputDirectory'] + _sep + self.cache[
                'id'] + _sep + f"fold_{self.cache['split_ix']}"
            _os.makedirs(self.cache['log_dir'], exist_ok=True)

            trainer.init_nn(init_weights=True)
            self.cache['current_nn_state'] = 'current.nn.pt'
            self.cache['best_nn_state'] = 'best.nn.pt'
            trainer.save_checkpoint(name=self.cache['current_nn_state'])
            trainer.load_data_indices(dataset_cls, split_key='train')
            nxt_phase = 'computation'

        if nxt_phase == 'computation':
            """ Train/validation and test phases """
            self.out.update(mode=self.input['global_modes'].get(self.state['clientId'], self.cache['mode']))

            trainer.init_nn(init_weights=False)
            trainer.load_checkpoint_from_key(key='current_nn_state')

            if self.input.get('save_current_as_best'):
                trainer.save_checkpoint(file_name=self.cache['best_nn_state'])
                self.cache['best_epoch'] = self.cache['epoch']

            if any(m == 'train' for m in self.input['global_modes'].values()):
                """
                All sites must begin/resume the training the same time.
                To enforce this, we have a 'val_waiting' status. Lagged sites will go to this status, 
                   and reshuffle the data,
                take part in the training with everybody until all sites go to 'val_waiting' status.
                """
                self.out.update(**trainer.train(dataset_cls))

            elif self.out['mode'] == 'validation':
                """
                Once all sites are in 'val_waiting' status, remote issues 'validation' signal.
                Once all sites run validation phase, they go to 'train_waiting' status.
                Once all sites are in this status, remote issues 'train' signal
                 and all sites reshuffle the indices and resume training.
                We send the confusion matrix to the remote to accumulate global score for model selection.
                """
                self.out.update(**trainer.validation(dataset_cls))

            elif self.out['mode'] == 'test':
                self.out.update(**trainer.test(dataset_cls))
                self.out['mode'] = self.cache['args']['mode']
                nxt_phase = 'next_run_waiting'

        elif nxt_phase == 'success':
            """ This phase receives global scores from the aggregator."""
            _shutil.copy(f"{self.state['baseDirectory']}{_sep}{self.input['results_zip']}.zip",
                         f"{self.state['outputDirectory'] + _sep + self.cache['id']}{_sep}{self.input['results_zip']}.zip")

        self.out['phase'] = nxt_phase

    def send(self):
        output = _json.dumps({'output': self.out, 'cache': self.cache})
        _sys.stdout.write(output)
