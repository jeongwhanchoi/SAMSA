import torch.nn.functional as F
import torch.nn as nn
import torch
import math

@torch.compile
def lily(q, k):
    _, _, _, d = q.shape
    q_, k_ = torch.unsqueeze(q, 3), torch.unsqueeze(k, 2)
    qk = torch.abs(q_ + k_) - torch.abs(q_ - k_)
    return torch.sum(qk, dim=-1) / math.sqrt(d)