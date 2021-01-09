import numpy as _np
import torch as _torch

grad_file_format = '.npy'
grad_precision_bit = 16

grads_numpy = grad_file_format == '.npy'
grads_torch = grad_file_format == '.tar'

