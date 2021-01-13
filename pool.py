from enum import Enum


class Modes(int, Enum):
    T = 1


import torch.distributed as _dist

print()
