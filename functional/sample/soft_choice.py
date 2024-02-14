import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_choice(x):
    x1 = F.softplus(x[:,:,0,:])
    x2 = F.softplus(x[:,:,1,:])
    x1.requires_grad = True
    x2.requires_grad = True
    xs = x1 + x2 + 1e-9
    return torch.stack([x1, x2], dim=2) / xs.unsqueeze(2), x1, x2