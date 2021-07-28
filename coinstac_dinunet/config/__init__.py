import torch as _torch
import sys as _sys
import random as _random
import argparse as _ap

grad_precision_bit = 16
grad_file_ext = '.npy'  # numpy
min_batch_size = 4

grads_file = f'grads{grad_file_ext}'
avg_grads_file = f'avg_grads{grad_file_ext}'
weights_file = 'weights.tar'

metrics_eps = 1e-5
metrics_num_precision = 5

score_delta = 0.0001
score_high = 1.0
score_low = 0.0

gpu_available = _torch.cuda.is_available()
num_gpus = _torch.cuda.device_count()

max_size = _sys.maxsize

current_seed = _random.randint(0, 2 ** 24)


def boolean_string(s):
    try:
        return str(s).strip().lower() == 'true'
    except:
        return False
