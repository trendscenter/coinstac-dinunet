## coinstac-dinunet
#### Distributed Neural Network implementation  on COINSTAC.

![PyPi version](https://img.shields.io/pypi/v/coinstac-dinunet)
[![YourActionName Actions Status](https://github.com/trendscenter/coinstac-dinunet/workflows/build/badge.svg)](https://github.com/trendscenter/coinstac-dinunet/actions)
![versions](https://img.shields.io/pypi/pyversions/pybadges.svg)

```
pip install coinstac-dinunet
```
#### Specify supported packages like pytorch & torchvision in a requirements.txt file
#### Highlights:
```
1. Handles multi-network/complex training schemes.
2. Automatic data splitting/k-fold cross validation.
3. Automatic model checkpointing.
4. GPU enabled local sites.
5. Customizable metrics(w/Auto serialization between nodes) to work with any schemes.
6. We can integrate any custom reduction and learning mechanism by extending coinstac_dinunet.distrib.reducer/learner.
7. Realtime profiling each sites by specifying in compspec file(see dinune_fsv example below for details). 
...
```


<hr />

![DINUNET](assets/dinunet.png)


### Working examples:
1. **[FreeSurfer volumes classification.](https://github.com/trendscenter/dinunet_implementations/)**
2. **[VBM 3D images classification.](https://github.com/trendscenter/dinunet_implementations_gpu)**

### [Running an analysis](https://github.com/trendscenter/coinstac-instructions/blob/master/coinstac-how-to-run-analysis.md) in the coinstac App.
### Add a new NN computation to COINSTAC (Development guide):
#### imports
```python
from coinstac_dinunet import COINNDataset, COINNTrainer, COINNLocal
from coinstac_dinunet.metrics import COINNAverages, Prf1a
```

#### 1. Define Data Loader
```python
class MyDataset(COINNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.labels = None

    def load_index(self, id, file):
        data_dir = self.path(id, 'data_dir') # data_dir comes from inputspecs.json
        ...
        self.indices.append([id, file])

    def __getitem__(self, ix):
        id, file = self.indices[ix]
        data_dir = self.path(id, 'data_dir') # data_dir comes from inputspecs.json
        label_dir = self.path(id, 'label_dir') # label_dir comes from inputspecs.json
        ...
        # Logic to load, transform single data item.
        ...
        return {'inputs':.., 'labels': ...}
```

#### 2. Define Trainer
```python
class MyTrainer(COINNTrainer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def _init_nn_model(self):
        self.nn['model'] = MYModel(in_size=self.cache['input_size'], out_size=self.cache['num_class'])

    def iteration(self, batch):
        inputs, labels = batch['inputs'].to(self.device['gpu']).float(), batch['labels'].to(self.device['gpu']).long()

        out = F.log_softmax(self.nn['model'](inputs), 1)
        loss = F.nll_loss(out, labels)
        _, predicted = torch.max(out, 1)
        score = self.new_metrics()
        score.add(predicted, labels)
        val = self.new_averages()
        val.add(loss.item(), len(inputs))
        return {'out': out, 'loss': loss, 'averages': val,
                'metrics': score, 'prediction': predicted}
```

#### 3. Add entries to:
* Local node entry point [CPU](https://github.com/trendscenter/dinunet_implementations/blob/master/local.py), [GPU](https://github.com/trendscenter/dinunet_implementations_gpu/blob/master/local.py)
* Aggregator node point [CPU](https://github.com/trendscenter/dinunet_implementations/blob/master/remote.py), [GPU](https://github.com/trendscenter/dinunet_implementations_gpu/blob/master/remote.py)
* compspec.json file [CPU](https://github.com/trendscenter/dinunet_implementations/blob/master/compspec.json), [GPU](https://github.com/trendscenter/dinunet_implementations_gpu/blob/master/compspec.json)

<hr />

#### Advanced use cases:

* **Define custom metrics:**
  - Extend [coinstac_dinunet.metrics.COINNMetrics](https://github.com/trendscenter/coinstac-dinunet/blob/main/coinstac_dinunet/metrics/metrics.py)
  - Example: [coinstac_dinunet.metrics.Prf1a](https://github.com/trendscenter/coinstac-dinunet/blob/main/coinstac_dinunet/metrics/metrics.py) for Precision, Recall, F1, and Accuracy

* **Define [custom DataHandle](https://github.com/trendscenter/dinunet_implementations/blob/8411bb95a0bef86bf6451b39f580f79c3c74eb94/comps/fs/__init__.py#L75)**
* **Define [Custom Learner](https://github.com/trendscenter/coinstac-dinunet/blob/main/coinstac_dinunet/distrib/learner.py) / [custom Aggregator](https://github.com/trendscenter/coinstac-dinunet/blob/main/coinstac_dinunet/distrib/reducer.py) (Default is Distributed SGD)**