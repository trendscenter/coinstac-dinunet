#!/usr/bin/python
import json
import os
import pandas as pd
import sys
import torch
import torch.nn.functional as F

from coinstac_dinunet import COINNTrainer, COINNDataset, COINNLocal
from .models import MSANNet

# import pydevd_pycharm
# pydevd_pycharm.settrace('172.17.0.1', port=8881, stdoutToServer=True, stderrToServer=True, suspend=False)


class FSDataset(COINNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)


    def load_indices(self, files=None, **kw):
        labels_file = os.listdir(self.label_dir)[0]
        labels = pd.read_csv(self.label_dir + os.sep + labels_file).set_index('freesurferfile')
        for file in files:
            y = labels.loc[file]['label']
            """
            int64 could not be json serializable.
            """
            self.indices.append([file, int(y)])

    def __getitem__(self, ix):
        file, y = self.indices[ix]
        df = pd.read_csv(self.data_dir + os.sep + file, sep='\t', names=['File', file], skiprows=1)
        df = df.set_index(df.columns[0])
        x = df.T.iloc[0].values
        return {'inputs': torch.tensor(x), 'labels': torch.tensor(y), 'ix': torch.tensor(ix)}


class FSTrainer(COINNTrainer):
    def _init_nn_model(self):
        self.nn['model'] = MSANNet(in_size=self.cache['input_size'], hidden_sizes=self.cache['hidden_sizes'],
                                   out_size=self.cache['num_class'])

    def iteration(self, batch):
        inputs = batch['inputs'].to(self.device['gpu']).float()
        labels = batch['labels'].to(self.device['gpu']).long()
        indices = batch['ix'].to(self.device['gpu']).long()

        first_model = list(self.nn.keys())[0]
        out = F.log_softmax(self.nn[first_model](inputs), 1)
        wt = torch.randint(1, 101, (2,)).to(self.device['gpu']).float()
        loss = F.nll_loss(out, labels, weight=wt)

        _, predicted = torch.max(out, 1)
        score = self.new_metrics()
        score.add(predicted, labels)
        val = self.new_averages()
        val.add(loss.item(), len(inputs))
        return {'out': out, 'loss': loss, 'avg_loss': val, 'score': score,
                'prediction': predicted, 'indices': indices}


if __name__ == "__main__":
    args = json.loads(sys.stdin.read())
    local = COINNLocal(**args)
    local.compute(FSDataset, FSTrainer)
    local.send()
