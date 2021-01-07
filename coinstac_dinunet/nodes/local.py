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

import coinstac_dinunet.data as _dt


class COINNLocal:
    def __init__(self, **kw):
        self.out = {}
        self.cache = kw['cache']
        self.input = kw['input']
        self.state = kw['state']

    def compute(self, dataset_cls, trainer_cls):
        trainer = trainer_cls(cache=self.cache, input=self.input, state=self.state)
        nxt_phase = self.input.get('phase', 'init_runs')
        if nxt_phase == 'init_runs':
            """
            Generate folds as specified.
            """
            self.cache.update(**self.input)
            self.out.update(**_dt.init_k_folds(self.cache, self.state))
            self.cache['_mode_'] = self.input['mode']
            self.out['mode'] = self.cache['mode']

        if nxt_phase == 'init_nn':
            """
            Initialize neural network/optimizer and GPUs
            """

            self.cache.update(**self.input['run'][self.state['clientId']], epoch=0, cursor=0, train_log=[])
            self.cache['split_file'] = self.cache['splits'][str(self.cache['split_ix'])]
            self.cache['log_dir'] = self.state['outputDirectory'] + _sep + self.cache[
                'id'] + _sep + f"fold_{self.cache['split_ix']}"
            _os.makedirs(self.cache['log_dir'], exist_ok=True)

            trainer.init_nn(init_weights=True)
            self.cache['current_nn_state'] = 'current.nn.pt'
            self.cache['best_nn_state'] = 'best.nn.pt'
            trainer.save_checkpoint(name=self.cache['current_nn_state'])

            dataset = dataset_cls(cache=self.cache, state=self.state, mode=self.cache['mode'])
            _dt.cache_data_indices(dataset, self.cache, self.cache.get('min_batch_size', 4))
            nxt_phase = 'computation'

        if nxt_phase == 'computation':
            """
            Train/validation and test phases
            """
            trainer.init_nn(init_weights=False)
            out_, nxt_phase = trainer.train_n_eval(dataset_cls, nxt_phase)
            self.out.update(**out_)

        elif nxt_phase == 'success':
            """
            This phase receives global scores from the aggregator.
            """
            _shutil.copy(f"{self.state['baseDirectory']}{_sep}{self.input['results_zip']}.zip",
                         f"{self.state['outputDirectory'] + _sep + self.cache['id']}{_sep}{self.input['results_zip']}.zip")

        self.out['phase'] = nxt_phase

    def send(self):
        output = _json.dumps({'output': self.out, 'cache': self.cache})
        _sys.stdout.write(output)
