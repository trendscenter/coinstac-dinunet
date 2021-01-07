import torch

"""
Common names used in local and master
folder names to save the log files of remote and local in their output dirs for a computation
"""
profile_log_dir_name = 'epoch_1_bs_4_cl_4_test'

"""
Format of grad files to be transmitted
"""
grad_file_format = '.pt'  # can also to '.npy' or '.tar'

is_format_numpy = grad_file_format == '.npy'
is_format_torch = grad_file_format == '.pt' or grad_file_format == '.tar'

assert (is_format_torch or is_format_numpy), "Allowed formats for transmission: .npy, .pt, .tar"

"""
File name used in transfer directory which has local gradients for an iteration
"""
grads_file = 'grads' + grad_file_format

"""
File name used in transfer directory which has local gradients for an iteration
"""
avg_grads_file = 'avg_grads' + grad_file_format

"""
float precision to be used for transmission
"""
float_precision = torch.float16
