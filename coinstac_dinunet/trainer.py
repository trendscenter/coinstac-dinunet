"""
@author: Aashis Khanal
@email: sraashis@gmail.com
"""
from abc import ABC
from os import sep as _sep

import coinstac_dinunet.config as _conf
from coinstac_dinunet.config.keys import *
from coinstac_dinunet.utils.utils import performance_improved_
from .nn import NNTrainer as _NNTrainer
import coinstac_dinunet.metrics as _metrics


class COINNTrainer(_NNTrainer, ABC):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.nn = self.cache.setdefault('nn', {})
        self.device = self.cache.setdefault('device', {})
        self.optimizer = self.cache.setdefault('optimizer', {})

    def _save_if_better(self, epoch, val_metrics):
        r"""
        Save the current model as best if it has better validation scores in pretraining step.
        """
        out = {}
        val_score = val_metrics.extract(self.cache['monitor_metric'])
        if performance_improved_(epoch, val_score, self.cache):
            out['weights_file'] = _conf.weights_file
            self.save_checkpoint(file_path=self.state['transferDirectory'] + _sep + out['weights_file'])
        return out

    def validation_distributed(self, dataset_cls):
        out = {}
        validation_dataset = self.data_handle.dataset.get('validation')
        if validation_dataset and not isinstance(validation_dataset, list):
            validation_dataset = [validation_dataset]

        if validation_dataset:
            avg, metrics = self.evaluation(
                mode='validation',
                save_pred=False,
                dataset_list=validation_dataset,
                use_padded_sampler=True
            )
            out[Key.VALIDATION_SERIALIZABLE] = [{'averages': avg.serialize(), 'metrics': metrics.serialize()}]
        self.cache['cursor'] = 0
        return out

    def test_distributed(self, dataset_cls):
        out = {}
        self.load_checkpoint(self.cache['log_dir'] + _sep + self.cache['best_nn_state'])
        test_dataset = self.data_handle.get_test_dataset(dataset_cls)
        if test_dataset and not isinstance(test_dataset, list):
            test_dataset = [test_dataset]

        if test_dataset:
            avg, metrics = self.evaluation(mode='test', save_pred=True,
                                           dataset_list=test_dataset)
            out[Key.TEST_SERIALIZABLE] = [{'averages': avg.serialize(), 'metrics': metrics.serialize()}]
        return out

    def set_monitor_metric(self):
        """Must be set from COINNLocal's constructor"""
        pass

    def set_log_headers(self):
        """Must be set from COINNLocal's constructor"""
        pass

    def new_metrics(self):
        if self.cache.get('num_class') == 2:
            if self.cache.get("monitor_metric") in ["precision", "recall", "accuracy", "overlap", 'f1']:
                return _metrics.Prf1a()

            elif self.cache.get("monitor_metric") == "auc":
                return _metrics.AUCROCMetrics()

        elif self.cache.get('num_class') > 2:
            return _metrics.ConfusionMatrix(num_classes=self.cache['num_class'])
