"""
@author: Aashis Khanal
@email: sraashis@gmail.com
"""

import abc as _abc
import time as _time
import typing as _typing

import numpy as _np
import torch as _torch
from sklearn import metrics as _metrics

from coinstac_dinunet.config import metrics_num_precision as _nump, metrics_eps as _eps


class COINNMetrics:
    def __init__(self, device='cpu', **kw):
        self.device = device

    @_abc.abstractmethod
    def add(self, *args, **kw):
        r"""
        Add two tensor to collect scores.
        Example implementation easytorch.utils.measurements.Prf1a().
        Calculate/store all True Positives, False Positives, True Negatives, False Negatives:
           out = F.softmax(core(x), 1)
           _, pred = torch.max(out, 1)
           sc = self.new_metrics()
           sc.add(pred, labels)
        """
        raise NotImplementedError('Must be implemented.')

    def accumulate(self, other):
        r"""
        Add all the content from another ETMetrics object.
        """
        pass

    def reset(self):
        r"""
        Clear all the content of self.
        """
        pass

    def get(self, *args, **kw) -> _typing.List[float]:
        r"""
        Computes/returns list of scores.
            Example: easytorch.utils.measurements.Prf1a() returns
            Precision, Recall, F1, Accuracy from the collected TP, TN, FP, FN.
        """
        return [0.0]

    @property
    def eps(self):
        r"""
        Epsilon(default 10e-5) for numerical stability.
        """
        return _eps

    @property
    def num_precision(self):
        r"""
        Numerical Precision(default 5) for nice looking numbers.
        """
        return _nump

    @property
    def time(self):
        return _time.time()

    def extract(self, name):
        sc = getattr(self, name.lower())
        if callable(sc):
            sc = sc()
        return sc

    @_abc.abstractmethod
    def serialize(self):
        pass

    @_abc.abstractmethod
    def reduce_sites(self, scores):
        pass


class COINNAverages(COINNMetrics):
    def __init__(self, num_averages=1, **kw):
        r"""
        This class can keep track of K averages.
        For example, in GAN we need to keep track of Generators loss
        """
        super().__init__(**kw)
        self.values = _np.array([0.0] * num_averages, dtype=_np.float)
        self.counts = _np.array([0.0] * num_averages, dtype=_np.float)
        self.num_averages = num_averages

    def add(self, val=0, n=1, index=0):
        r"""
        Keep adding val, n to get the average later.
        Index is the position on where to add the values.
        For example:
            avg = ETAverages(num_averages=2)
            avg.add(lossG.item(), len(batch), 0)
            avg.add(lossD.item(), len(batch), 1)
        """
        self.values[index] += val * n
        self.counts[index] += n

    def accumulate(self, other):
        r"""
        Add another ETAverage object to self
        """
        self.values += other.values
        self.counts += other.counts

    def reset(self):
        r"""
        Clear all the content of self.
        """
        self.values = _np.array([0.0] * self.num_averages)
        self.counts = _np.array([0.0] * self.num_averages)

    def get(self) -> _typing.List[float]:
        r"""
        Computes/Returns self.num_averages number of averages in vectorized way.
        """
        counts = self.counts.copy()
        counts[counts == 0] = _np.inf
        return _np.round(self.values / counts, self.num_precision)

    def average(self, reduce_mean=True):
        avgs = self.get()
        if reduce_mean:
            return round(sum(avgs) / len(avgs), self.num_precision)
        return avgs

    def serialize(self):
        return [self.values.tolist(), self.counts.tolist()]

    def reduce_sites(self, scores: list):
        self.values, self.counts = _np.array(scores).sum(0)


class Prf1a(COINNMetrics):
    r"""
    A class that has GPU based computation of:
        Precision, Recall, F1 Score, Accuracy, and Overlap(IOU).
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.tn, self.fp, self.fn, self.tp = 0, 0, 0, 0
        self._precision = 0
        self._recall = 0
        self._accuracy = 0

    def add(self, pred, true):
        y_true = true.clone().int().view(1, -1).squeeze()
        y_pred = pred.clone().int().view(1, -1).squeeze()

        y_true[y_true == 255] = 1
        y_pred[y_pred == 255] = 1

        y_true = y_true * 2
        y_cases = y_true + y_pred
        self.tp += _torch.sum(y_cases == 3).item()
        self.fp += _torch.sum(y_cases == 1).item()
        self.tn += _torch.sum(y_cases == 0).item()
        self.fn += _torch.sum(y_cases == 2).item()

    def accumulate(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.tn += other.tn
        self.fn += other.fn

    def reset(self):
        self.tn, self.fp, self.fn, self.tp = [0] * 4

    @property
    def precision(self):
        p = self.tp / max(self.tp + self.fp, self.eps)
        return round(max(p, self._precision), self.num_precision)

    @property
    def recall(self):
        r = self.tp / max(self.tp + self.fn, self.eps)
        return round(max(r, self._recall), self.num_precision)

    @property
    def accuracy(self):
        a = (self.tp + self.tn) / \
            max(self.tp + self.fp + self.fn + self.tn, self.eps)
        return round(max(a, self._accuracy), self.num_precision)

    @property
    def f1(self):
        return self.f_beta(beta=1)

    def f_beta(self, beta=1):
        f_beta = (1 + beta ** 2) * self.precision * self.recall / \
                 max(((beta ** 2) * self.precision) + self.recall, self.eps)
        return round(f_beta, self.num_precision)

    def get(self):
        return [self.accuracy, self.f1, self.precision, self.recall]

    @property
    def overlap(self):
        o = self.tp / max(self.tp + self.fp + self.fn, self.eps)
        return round(o, self.num_precision)

    def serialize(self):
        return [self.accuracy, self.precision, self.recall]

    def reduce_sites(self, scores: list):
        self._accuracy, self._precision, self._recall = _np.array(scores).mean(0)


class ConfusionMatrix(COINNMetrics):
    """
    Confusion matrix  is used in multi class classification case.
    x-axis is predicted. y-axis is true label.
    F1 score from average precision and recall is calculated
    """

    def __init__(self, num_classes=None, device='cpu', **kw):
        super().__init__(device, **kw)
        self.num_classes = num_classes
        self.matrix = _torch.zeros(num_classes, num_classes).float()
        self._precision = [0] * self.num_classes
        self._recall = [0] * self.num_classes
        self._accuracy = 0
        self.device = device

    def reset(self):
        self.matrix = _torch.zeros(self.num_classes, self.num_classes).float()

    def accumulate(self, other: COINNMetrics):
        self.matrix += other.matrix

    def add(self, pred: _torch.Tensor, true: _torch.Tensor):
        pred = pred.clone().long().reshape(1, -1).squeeze()
        true = true.clone().long().reshape(1, -1).squeeze()
        self.matrix += _torch.sparse.LongTensor(
            _torch.stack([pred, true]).to(self.device),
            _torch.ones_like(pred).long().to(self.device),
            _torch.Size([self.num_classes, self.num_classes])).to_dense().to(self.device)

    def precision(self, average=True):
        precision = [0] * self.num_classes
        for i in range(self.num_classes):
            _p = max(self._precision[i], self.eps)
            p = max(_torch.sum(self.matrix[:, i]).item(), self.eps)
            precision[i] = self.matrix[i, i] / max(p, _p)

        return sum(precision) / self.num_classes if average else precision

    def recall(self, average=True):
        recall = [0] * self.num_classes
        for i in range(self.num_classes):
            _r = max(self._recall[i], self.eps)
            r = max(_torch.sum(self.matrix[i, :]).item(), self.eps)
            recall[i] = self.matrix[i, i] / max(r, _r)

        return sum(recall) / self.num_classes if average else recall

    def f1(self, average=True):
        f_1 = []
        precision = [self.precision(average)] if average else self.precision(average)
        recall = [self.recall(average)] if average else self.recall(average)
        for p, r in zip(precision, recall):
            f_1.append(2 * p * r / max(p + r, self.eps))
        f_1 = _np.array(f_1)
        return f_1[0] if average else f_1

    def accuracy(self):
        return max(self.matrix.trace().item() / max(self.matrix.sum().item(), self.eps), self._accuracy)

    def get(self):
        return [round(self.accuracy(), self.num_precision), round(self.f1(), self.num_precision),
                round(self.precision(), self.num_precision), round(self.recall(), self.num_precision)]

    def serialize(self, **kw):
        return [self.accuracy(), self.precision().tolist(), self.recall().tolist()]

    def reduce_sites(self, scores: list):
        self._accuracy, self._precision, self._recall = _np.array(scores).mean(0)


class AUCROCMetrics(COINNMetrics):
    __doc__ = "Restricted to binary case"

    def __init__(self, device='cpu', **kw):
        super().__init__(device, **kw)
        self.probabilities = []
        self.labels = []
        self.fpr = None
        self.tpr = None
        self.thresholds = None
        self._auc = 0

    def accumulate(self, other):
        self.probabilities += other.probabilities
        self.labels += other.labels

    def reset(self):
        self.probabilities = []
        self.labels = []

    def auc(self):
        if self._auc <= 0 and len(self.labels) > 0:
            self.fpr, self.tpr, self.thresholds = _metrics.roc_curve(self.labels, self.probabilities, pos_label=1)
            return _metrics.auc(self.fpr, self.tpr)
        return self._auc

    def get(self, *args, **kw):
        return [round(self.auc(), self.num_precision)]

    def add(self, pred: _torch.Tensor, true: _torch.Tensor):
        self.probabilities += pred.flatten().clone().detach().cpu().tolist()
        self.labels += true.clone().flatten().detach().cpu().tolist()

    def serialize(self):
        return [self.auc()]

    def reduce_sites(self, scores: list):
        self._auc = _np.array(scores).mean()
