import numpy as _np
import torch as _torch

grad_file_format = '.npy'
grad_precision_bit = 16

grads_numpy = grad_file_format == '.npy'
grads_torch = grad_file_format == '.tar'


metrics_eps = 10e-5
metrics_num_precision = 5

gpu_available = _torch.cuda.is_available()
num_gpus = _torch.cuda.device_count()
