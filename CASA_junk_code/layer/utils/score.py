import torch.nn.functional as F
import torch.nn as nn
import torch
from layer.utils.gumbel_rao import GRSoftmax
import math

def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(2).unsqueeze(2)
    y_t = torch.transpose(y, 1, 2)
    y_norm = (y**2).sum(2).unsqueeze(1)

    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    return dist 

class euclidean_score(nn.Module):
    def __init__(self, norm, nhead):
        super().__init__()
        if norm == True:
            self.norm = HeadValueNormalization(nhead)
        else:
            self.norm = nn.Identity()
        self.r = nn.Parameter(torch.ones(1, requires_grad=True), requires_grad=True)

    def forward(self, q, k):
        batch_size, head, length_k, d_tensor = k.size()
        _, _, length_q, _ = q.size()
        dist = pairwise_distances(q.reshape(batch_size * head, length_q, d_tensor), 
                                    k.reshape(batch_size * head, length_k, d_tensor)).reshape(batch_size, head, length_q, length_k)
        dist = (dist / d_tensor) * -1
        dist = self.norm(dist)
        dist = F.tanh(dist) * F.softplus(self.r)
        return dist

class scale_dot_score(nn.Module):
    def __init__(self, norm, nhead):
        super().__init__()
        if norm == True:
            self.norm = HeadValueNormalization(nhead)
        else:
            self.norm = nn.Identity()

    def forward(self, q, k):
        batch_size, head, length_k, d_tensor = k.size()
        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)
        return self.norm(score)

class composite_score(nn.Module):
    def __init__(self, norm, nhead):
        super().__init__()
        if norm == True:
            self.norm = HeadValueNormalization(nhead)
        else:
            self.norm = nn.Identity()

    def forward(self, q, k, z=None):
        _, _, _, d = q.shape
        msd = math.sqrt(d)
        q = torch.sum(q, dim=-1, keepdim=True) / msd
        k = torch.sum(k, dim=-1, keepdim=True) / msd
        q_, k_ = torch.unsqueeze(q, 3), torch.unsqueeze(k, 2)
        qk = torch.cat([q_ + torch.zeros_like(k_), k_ + torch.zeros_like(q_)], dim=-1)
        if z != None:
            qk = qk + z
        return torch.max(qk, dim=-1)[0]
        # qk = F.softmax(qk, dim=-1) * qk
        # return torch.sum(qk, dim=-1)

class HeadValueNormalization(nn.Module):
    def __init__(self, nhead, momentum=0.9, epsilon=1e-5):
        super(HeadValueNormalization, self).__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = nn.Parameter(torch.zeros(1, nhead, 1, 1), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(1, nhead, 1, 1), requires_grad=False)
        self.is_initialized = False  # Add a flag to check initialization
        self.w = nn.Parameter(torch.randn(1, nhead, 1, 1, requires_grad=True), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1, nhead, 1, 1, requires_grad=True), requires_grad=True)

    def forward(self, x):
        if self.training:
            batch_mean = torch.mean(x, dim=(0, 2, 3), keepdims=True)
            batch_var = torch.var(x, dim=(0, 2, 3), keepdims=True)

            if not self.is_initialized:
                self.running_mean.data = batch_mean
                self.running_var.data = batch_var
                self.is_initialized = True
            else:
                self.running_mean.data = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
                self.running_var.data = self.momentum * self.running_var + (1 - self.momentum) * batch_var

        normalized_x = (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)
        return normalized_x * self.w + self.b