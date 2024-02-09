import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.utils import GRSoftmax
import random
import math
from layer.utils import Copy

class GSoftmax(nn.Module):
    def __init__(self, initial_temp=5.0, dec_rate=0.999):
        super(GSoftmax, self).__init__()
        self.initial_temp = initial_temp
        self.dec_rate = dec_rate

    def forward(self, x, dim):
        if self.training == True:
            if self.initial_temp >= 0.05:
                self.initial_temp *= self.dec_rate
        return F.gumbel_softmax(x, tau=self.initial_temp, hard=False)

class SAM(nn.Module):
    def __init__(self,
                 n_sampled_points_lb,
                 n_sampled_points_ub,
                 n_center,
                 d_model: int,
                 drop_point: float = 0.1):
        super(SAM, self).__init__()
        self.n_sampled_points_lb = n_sampled_points_lb
        self.n_sampled_points_ub = n_sampled_points_ub
        self.n_center = n_center
        self.d_model = d_model
        self.pointwise = nn.Sequential(nn.LayerNorm(d_model),
                                       nn.LazyLinear(n_center))
        self.drop_point =  1 - drop_point
        self.GBR = GRSoftmax(temp=0.1, k=3)

    def forward(self, x, mask=None):
        if self.training == True:
            n_sampled_points = random.randint(self.n_sampled_points_lb, self.n_sampled_points_ub)
        else:
            n_sampled_points = self.n_sampled_points_ub
        x_fwd = self.pointwise(x) / math.sqrt(2)

        if self.training == True:
            drop_ = torch.bernoulli(torch.ones(x.shape[0], x.shape[1], device=x.device) * self.drop_point).unsqueeze(dim=2)
            x_fwd = x_fwd + (1 -  drop_) * (-1e9)
        else:
            pass
        if mask is not None:
            mask = mask.unsqueeze(dim=2)
            x_fwd = x_fwd + mask * (-1e9)
        x_fwd = Copy.apply(x_fwd, n_sampled_points)
        p_fwd = self.GBR(x_fwd, dim=1)
        p_fwd = torch.transpose(p_fwd, 1, 2)

        result = torch.bmm(p_fwd, x)
        return result, p_fwd