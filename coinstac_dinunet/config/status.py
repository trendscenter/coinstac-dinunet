from enum import Enum as _Enum


class Phase(str, _Enum):
    INIT_RUNS = 'init_runs'
    INIT_NN = 'init_nn'
    COMPUTATION = 'computation'
    NEXT_RUN_WAITING = 'next_run_waiting'
    SUCCESS = 'success'


class Mode(str, _Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'
    VALIDATION_WAITING = 'validation_waiting'
    TRAIN_WAITING = 'train_waiting'
