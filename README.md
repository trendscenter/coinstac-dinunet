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

### General use case:

#### 1. Define Data Loader
```python

```

#### 2. Define Local Node
```python

```
#### 3. Define Remote Node
```python

```
#### 4. Define Trainer
```python

```

#### 5. Define custom metrics
```python

```

### Default arguments:
    Directly passed parameters in coinstac_dinunet.nodes.COINNLocal, args passed through inputspec will override in the same order.
* ***mode***: str = None, Eg. train/test
* ***batch_size***: int = 4 
* ***epochs***: int = 21
* ***learning_rate***: float = 0.001
* ***gpus***: _List[int] = None, Eg. [0], [1], [0, 1]...
* ***pin_memory***: bool = True, if cuda available
* ***num_workers***: int = 0
* ***load_limit***: int = float('inf'), Limit on dataset to load for debugging puprose.
* ***pretrained_path***: str = None, Path to pretrained weights
* ***patience***: int = 5, patience to end training by monitoring validation scores.
* ***load_sparse***: bool = False, Load each data item in separate order to reconstruct images from patches, if needed.
* ***num_folds***: int = None, Number of k-folds. 
* ***split_ratio***: _List[float] = (0.6, 0.2, 0.2), Exclusive to num_folds.
```
Custom data splits can be provied in path specified by split_dir for each sites in their respective inputspecs file.
```

