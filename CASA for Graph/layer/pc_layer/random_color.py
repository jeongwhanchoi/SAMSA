import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

class RandomColor(nn.Module):
    def __init__(self):
        super().__init__()
        self.dirichlet = Dirichlet(torch.tensor([1/3, 1/3, 1/3]))

    def forward(self, x, mask=None):
        x_ = x[:,:,3:6]
        b = x.shape[0]
        r = self.dirichlet.sample_n(b * 3).reshape(b, 1, 3, 3)
        r = r.to(x.device)
        x_ = (x_.unsqueeze(2) @ r).squeeze(2)
        x[:,:,3:6] = x_
        return x