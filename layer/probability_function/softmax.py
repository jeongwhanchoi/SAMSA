import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax(x):
    maxes = torch.max(x, 3, keepdim=True)
    shifted_x = x - maxes
    exp_shifted_x = torch.exp(shifted_x)
    partition_sum = torch.sum(exp_shifted_x, 3, keepdim=True)
    return exp_shifted_x / partition_sum