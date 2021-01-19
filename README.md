## coinstac-dinunet
#### Distributed Neural Network implementation  on COINSTAC.

[![YourActionName Actions Status](https://github.com/trendscenter/coinstac-dinunet/workflows/build/badge.svg)](https://github.com/trendscenter/coinstac-dinunet/actions)
![versions](https://img.shields.io/pypi/pyversions/pybadges.svg)

```
pip install coinstac-dinunet
```
#### Install supported pytorch & torchvision binaries in your device/docker ecosystem:
```
torch==1.5.1+cu92
torchvision==0.6.1+cu92
```

#### Highlights:
```
1. Handles multi-network/complex training schemes.
2. Automatic data splitting/k-fold cross validation.
3. Automatic model checkpointing.
4. GPU enabled local sites.
5. Customizable metrics(w/Auto serialization between nodes) to work with any schemes.
...
```


<hr />

## Pipeline for reducing gradients across sites.

![DINUNET](assets/dinunet.png)

### Full working examples
1. **[FreeSurfer volumes classification.](https://github.com/trendscenter/dinunet/tree/packaging)**
2. **[VBM 3D images classification.](https://github.com/trendscenter/dinunet_vbm)**
### General use case:
#### imports
```python
from coinstac_dinunet import COINNDataset, COINNTrainer, COINNRemote, COINNLocal
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
#### 3. Supply to local node in local.py
```python
if __name__ == "__main__":
    args = json.loads(sys.stdin.read())
    local = COINNLocal(cache=args['cache'], input=args['input'], state=args['state'])
    local.compute(MyDataset, MyTrainer)
    local.send()
```
#### 4. Define remote node in remote.py

```python
class MyRemote(COINNRemote):

    def _new_metrics(self):  #
        return coinstac_dinunet.metrics.Prf1a()

    def _new_averages(self):
        return coinstac_dinunet.metrics.COINNAverages()

    def _monitor_metric(self):
        return 'f1', 'maximize'


if __name__ == "__main__":
    args = json.loads(sys.stdin.read())
    remote = MyRemote(cache=args['cache'], input=args['input'], state=args['state'])
    remote.compute()
    remote.send()
```

#### Define custom metrics

- **Extend [coinstac_dinunet.metrics.COINNMetrics](https://github.com/trendscenter/coinstac-dinunet/blob/main/coinstac_dinunet/metrics/metrics.py)**
- **Example: [coinstac_dinunet.metrics.Prf1a](https://github.com/trendscenter/coinstac-dinunet/blob/main/coinstac_dinunet/metrics/metrics.py) for Precision, Recall, F1, and Accuracy**


### Default arguments:
* ***task_name***: str = None, Name of the task. [Required]
* ***mode***: str = None, Eg. train/test [Required]
* ***batch_size***: int = 4 
* ***epochs***: int = 21
* ***learning_rate***: float = 0.001
* ***gpus***: _List[int] = None, Eg. [0], [1], [0, 1]...
* ***pin_memory***: bool = True, if cuda available
* ***num_workers***: int = 0
* ***load_limit***: int = float('inf'), Limit on dataset to load for debugging purpose.
* ***pretrained_path***: str = None, Path to pretrained weights
* ***patience***: int = 5, patience to end training by monitoring validation scores.
* ***load_sparse***: bool = False, Load each data item in separate loader to reconstruct images from patches, if needed.
* ***num_folds***: int = None, Number of k-folds. 
* ***split_ratio***: _List[float] = (0.6, 0.2, 0.2), Exclusive to num_folds. 
  
**Directly passed parameters in coinstac_dinunet.nodes.COINNLocal, args passed through inputspec will override the defaults in the same order.**

**Custom data splits can be provided in the path specified by split_dir for each sites in their respective inputspecs file. This is mutually exclusive to both num_folds and split_ratio.**

<hr >


