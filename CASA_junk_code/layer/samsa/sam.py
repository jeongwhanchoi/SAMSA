import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import random
import math

class AlphaChoice(nn.Module):
    def __init__(self, alpha=0.95):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, dim):
        x1 = F.softplus(x[:,:,0,:])
        x2 = F.softplus(x[:,:,1,:])
        xs = x1 + x2 + 1e-9
        # return torch.stack([x1, x2], dim=2) / xs.unsqueeze(2), torch.mean(x1) + torch.mean(x2)
        return torch.stack([x1, x2], dim=2) / xs.unsqueeze(2), torch.mean(x2 / xs)
        # x1 = x[:,:,0,:]
        # x2 = x[:,:,1,:]
        # x = torch.stack([x1, x2], dim=2)
        # return F.gumbel_softmax(x, dim=2, hard=True, tau=0.1)

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def shufflerow(tensor, axis):
    row_perm = torch.rand(tensor.shape[:axis+1], device=tensor.device).argsort(axis)  # get permutation indices
    for _ in range(tensor.ndim-axis-1): row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(axis+1)], *(tensor.shape[axis+1:]))  # reformat this for the gather operation
    return tensor.gather(axis, row_perm)

def sample(x, x_score, k, sampler):
    b, ns, nd = x.shape
    x_score, indices = torch.sort(x_score, dim=1, descending=True)
    indices = indices.squeeze(-1)
    x = batched_index_select(x, dim=1, index=indices)
    x_top, x_score_top = x[:,:k,:], x_score[:,:k,:]
    x_bottom, x_score_bottom = x[:,k:,:], x_score[:,k:,:]
    x_bottom = torch.cat([x_bottom, x_score_bottom], dim=2)
    x_bottom = shufflerow(x_bottom, axis=1)
    x_bottom, x_score_bottom = x_bottom[:,:,:-1], x_bottom[:,:,-1].unsqueeze(-1)
    c = k
    x_bottom, x_score_bottom = x_bottom[:,:c,:].reshape(b, k, c // k, nd), x_score_bottom[:,:c,:].reshape(b, k, c // k, 1)
    x_top, x_score_top = x_top.unsqueeze(2), x_score_top.unsqueeze(2)
    x, x_score = torch.cat([x_top, x_bottom], dim=2), torch.cat([x_score_top, x_score_bottom], dim=2)
    x_score, regularization = sampler(x_score, dim=2)
    x = torch.sum(x * x_score, dim=2)
    return x, regularization

class SAM_(nn.Module):
    def __init__(self, 
                 n_sampled_points_lb,
                 n_sampled_points_ub,
                 d_model,
                 drop_point: float = 0.1):
        super().__init__()
        self.n_sampled_points_lb = n_sampled_points_lb
        self.n_sampled_points_ub = n_sampled_points_ub
        self.d_model = d_model
        self.pointwise = nn.Sequential(nn.LayerNorm(d_model),
                                       nn.LazyLinear(1))
        self.drop_point = 1 - drop_point
        self.sample = AlphaChoice()

    def forward(self, x, v=None, mask=None):
        x_fwd = self.pointwise(x) / math.sqrt(2)
        x_fwd += -1 * torch.log(-1 * torch.log((torch.rand_like(x_fwd) + 1e-20)) + 1e-20)
        if mask is not None:
            mask = mask.unsqueeze(dim=2)
            mask = mask[:,:x_fwd.shape[1],:]
            x_fwd = x_fwd + mask * (-1e9)

        n_sampled_points = random.randint(self.n_sampled_points_lb, self.n_sampled_points_ub)
        drop_ = torch.bernoulli(torch.ones(x.shape[0], x.shape[1], device=x.device) * self.drop_point).unsqueeze(dim=2)
        x_fwd = x_fwd + (1 -  drop_) * (-1e9)
        if v != None:
            x = torch.cat([x, v], dim=-1)
        return sample(x, x_fwd, n_sampled_points, self.sample)

class SAM(nn.Module):
    def __init__(self, 
                 n_sampled_points_lb,
                 n_sampled_points_ub,
                 d_model,
                 drop_point: float = 0.1):
        super().__init__()
        self.layers = SAM_(n_sampled_points_lb, 
                           n_sampled_points_ub,
                           d_model,
                           drop_point)

    def forward(self, x, v=None, mask=None):
        return self.layers(x, v, mask)