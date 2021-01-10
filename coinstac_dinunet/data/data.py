"""
@author: Aashis Khanal
@email: sraashis@gmail.com
@ref: https://github.com/sraashis/easytorch
"""

from torch.utils.data import DataLoader as _DataLoader, Dataset as _Dataset
from torch.utils.data._utils.collate import default_collate as _default_collate


def safe_collate(batch):
    r"""
    Savely select batches/skip errors in file loading.
    """
    return _default_collate([b for b in batch if b])


class COINNDataLoader(_DataLoader):

    def __init__(self, **kw):
        super(COINNDataLoader, self).__init__(**kw)

    @classmethod
    def new(cls, **kw):
        _kw = {
            'dataset': None,
            'batch_size': 1,
            'sampler': None,
            'shuffle': False,
            'batch_sampler': None,
            'num_workers': 0,
            'pin_memory': False,
            'drop_last': False,
            'timeout': 0,
            'worker_init_fn': None
        }
        for k in _kw.keys():
            _kw[k] = kw.get(k, _kw.get(k))
        return cls(collate_fn=safe_collate, **_kw)


class COINNDataset(_Dataset):
    def __init__(self, mode='init', limit=float('inf')):
        self.mode = mode
        self.limit = limit
        self.dataspecs = {}
        self.args = {}
        self.indices = []

    def load_index(self, id, file):
        r"""
        Logic to load indices of a single file.
        -Sometimes one image can have multiple indices like U-net where we have to get multiple patches of images.
        """
        self.indices.append([id, file])

    def _load_indices(self, id, files, **kw):
        r"""
        We load the proper indices/names(whatever is called) of the files in order to prepare minibatches.
        Only load lim numbr of files so that it is easer to debug(Default is infinite, -lim/--load-lim argument).
        """
        for file in files:
            if len(self) >= self.limit:
                break
            self.load_index(id, file)

        if kw.get('verbose', True):
            print(f'{id}, {self.mode}, {len(self)} Indices Loaded')

    def __getitem__(self, index):
        r"""
        Logic to load one file and send to model. The mini-batch generation will be handled by Dataloader.
        Here we just need to write logic to deal with single file.
        """
        raise NotImplementedError('Must be implemented by child class.')

    def __len__(self):
        return len(self.indices)

    def transforms(self, **kw):
        return None

    def add(self, files, cache: dict = None, state: dict = None):
        self.dataspecs[state['clientId']] = state
        self.args[state['clientId']] = cache['args']
        self._load_indices(id=state['clientId'], files=files, verbose=False)
