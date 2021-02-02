import torch as _torch
import sys as _sys

grad_precision_bit = 16
grad_file_ext = '.npy'  # numpy

grads_file = f'grads{grad_file_ext}'
avg_grads_file = f'avg_grads{grad_file_ext}'
weights_file = 'weights.tar'
metrics_eps = 10e-5
metrics_num_precision = 5
et_version = 2.4

gpu_available = _torch.cuda.is_available()
num_gpus = _torch.cuda.device_count()

data_load_lim = _sys.maxsize
