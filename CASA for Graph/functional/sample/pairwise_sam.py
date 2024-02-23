import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import random
import math
from functional.sample.sort_sample import sort_sample

class PairwiseSAM(nn.Module):
    def __init__(self, 
                 n_sampled_points,
                 d_model,
                 drop_point: float = 0.1):
        super().__init__()
        self.n_sampled_points = n_sampled_points
        self.d_model = d_model
        self.pointwise = nn.Sequential(nn.LayerNorm(d_model),
                                       nn.LazyLinear(1))
        self.drop_point = 1 - drop_point

    def forward(self, x, v=None, mask=None):
        x_fwd = self.pointwise(x) / math.sqrt(2)
        x_fwd += -1 * torch.log(-1 * torch.log((torch.rand_like(x_fwd) + 1e-20)) + 1e-20)
        if mask is not None:
            mask = mask.unsqueeze(dim=2)
            mask = mask[:,:x_fwd.shape[1],:]
            x_fwd = x_fwd + mask * (-1e9)

        n_sampled_points = self.n_sampled_points
        drop_ = torch.bernoulli(torch.ones(x.shape[0], x.shape[1], device=x.device) * self.drop_point).unsqueeze(dim=2)
        x_fwd = x_fwd + (1 -  drop_) * (-1e9)
        if v != None:
            x = torch.cat([x, v], dim=-1)
        return sort_sample(x, x_fwd, n_sampled_points)