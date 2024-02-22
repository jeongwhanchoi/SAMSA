import torch.nn.functional as F
import torch.nn as nn
import torch
import math

def maxout(q, k):
    _, _, _, d = q.shape
    q_, k_ = torch.unsqueeze(q, 3), torch.unsqueeze(k, 2)
    qk = torch.stack([q_ + torch.zeros_like(k_), k_ + torch.zeros_like(q_)], dim=-1)
    qk = (torch.max(qk, dim=-1)[0] - 0.5641895835) / 0.8256452711
    return torch.sum(qk, dim=-1) / math.sqrt(d)