import glob as _glob
import os as _os
import shutil as _shu
from typing import Tuple as _Tuple

import numpy as _np
import torch as _torch
from coinstac_dinunet.config.keys import Mode

from coinstac_dinunet.distrib.learner import COINNLearner
from coinstac_dinunet.distrib.reducer import COINNReducer


def check(logic, k, v, kw):
    phases = []
    for site_vars in kw.values():
        phases.append(site_vars.get(k) == v)
    return logic(phases)
