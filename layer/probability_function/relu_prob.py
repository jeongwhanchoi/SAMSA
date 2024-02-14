import torch
import torch.nn as nn
import torch.nn.functional as F

def relu_prob(x):
    x = F.relu(x)
    detr = torch.sum(x, dim=3, keepdim=True)
    return x / (detr + 1e-9)

def leaky_relu_prob(x, c):
    x = F.relu(x)
    detr = torch.sum(x, dim=3, keepdim=True) + torch.sum(F.softplus(c), dim=2, keepdim=True)
    return x / (detr + 1e-9)
