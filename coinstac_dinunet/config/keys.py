from enum import Enum as _Enum

SHARED_CACHE = ['computation_id', 'mode', 'epochs', 'num_folds', 'patience', 'score_delta', 'validation_epochs']


class Phase(str, _Enum):
    INIT_RUNS = 'init_runs'
    NEXT_RUN = 'next_run'
    PRE_COMPUTATION = 'pre_computation'
    COMPUTATION = 'computation'
    NEXT_RUN_WAITING = 'next_run_waiting'
    SUCCESS = 'success'


class Mode(str, _Enum):
    PRE_TRAIN = 'pre_train'
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'
    VALIDATION_WAITING = 'validation_waiting'
    TRAIN_WAITING = 'train_waiting'


class Key(str, _Enum):
    TRAIN_LOG = 'train_log'
    TRAIN_METRICS = 'train_metrics'
    TRAIN_SERIALIZABLE = 'serializable_train_scores'

    VALIDATION_LOG = 'validation_log'
    VALIDATION_METRICS = "validation_metrics"
    VALIDATION_SERIALIZABLE = 'serializable_validation_scores'

    TEST_LOG = 'test_log'
    TEST_METRICS = 'test_metrics'
    TEST_SERIALIZABLE = 'serializable_test_scores'

    GLOBAL_TEST_LOG = 'global_test_log'
    GLOBAL_TEST_METRICS = 'global_test_metrics'
    GLOBAL_TEST_SERIALIZABLE = 'serializable_global_test_scores'
