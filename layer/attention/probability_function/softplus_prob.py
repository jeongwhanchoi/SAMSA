import torch
import torch.nn as nn
import torch.nn.functional as F

def softplus_prob(x):
    x = F.softplus(x)
    detr = torch.sum(x, dim=3, keepdim=True)
    return x / (detr + 1e-9)
softplus_prob.is_leaky = False

def leaky_softplus_prob(x, c):
    x = F.softplus(x)
    detr = torch.sum(x, dim=3, keepdim=True) + torch.sum(F.softplus(c), dim=2, keepdim=True)
    return x / (detr + 1e-9)
leaky_softplus_prob.is_leaky = True
