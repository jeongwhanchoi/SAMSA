import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax(x):
    return F.softmax(x, dim=3)
softmax.is_leaky = False

def leaky_softmax(x, c): # x(b,h,s1,s2), c(b,h,s1,1)
    x = torch.cat([x, c], dim=3)
    maxes = torch.max(x, 3, keepdim=True)
    shifted_x = x - maxes
    exp_shifted_x = torch.exp(shifted_x)
    partition_sum = torch.sum(exp_shifted_x, 3, keepdim=True)
    ret = exp_shifted_x / partition_sum
    return ret[:,:,:,:ret.shape[3] - 1]
leaky_softmax.is_leaky = True
