import torch as _torch

grad_file_format = '.npy'
grad_precision_bit = 16

grads_numpy = grad_file_format == '.npy'
grads_torch = grad_file_format == '.tar'

grads_file = f'grads.{grad_file_format}'
avg_grads_file = f'avg_grads.{grad_file_format}'

metrics_eps = 10e-5
metrics_num_precision = 5
et_version = 2.4

gpu_available = _torch.cuda.is_available()
num_gpus = _torch.cuda.device_count()
