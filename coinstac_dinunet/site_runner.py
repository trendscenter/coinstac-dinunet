import json as json
import os

from coinstac_dinunet import COINNLocal
from coinstac_dinunet.config.keys import *


class SiteRunner(COINNLocal):
    def __init__(self, taks_id, data_path='test', site_index=0, **kw):
        cache = {}
        input = {}
        state = {}
        spec = json.loads(open(data_path + os.sep + "inputspec.json").read())[site_index]
        for k, v in spec.items():
            cache[k] = v['value']

        state["baseDirectory"] = f"{data_path}{os.sep}input{os.sep}" \
                                 f"local{site_index}{os.sep}simulatorRun"
        state["outputDirectory"] = f"{data_path}{os.sep}output{os.sep}" \
                                   f"local{site_index}{os.sep}simulatorRun{os.sep}_srun_{taks_id}"
        state["transferDirectory"] = state['outputDirectory']

        os.makedirs(state['outputDirectory'], exist_ok=True)
        state['clientId'] = f"local{site_index}"

        super().__init__(task_id=taks_id, cache=cache, input=input, state=state, **kw)

    def run(self, trainer_cls, dataset_cls, datahandle_cls, **kw):
        self.cache.update(**kw)

        self.input = {**self.input}
        self.input['phase'] = Phase.INIT_RUNS
        self.compute(None, trainer_cls, dataset_cls, datahandle_cls)

        self.cache['verbose'] = True
        self.input = {**self.input}
        self.input['phase'] = Phase.NEXT_RUN
        self.input['global_runs'] = {
            self.state['clientId']: {
                'split_ix': '0',
                'seed': 1,
                'pretrain': True}
        }
        self._pretrain_args = {**self.cache}
        self.compute(None, trainer_cls, dataset_cls, datahandle_cls)
