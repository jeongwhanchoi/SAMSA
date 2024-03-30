import torch
import torch.nn as nn
import torch.nn.functional as F

class pairwise_hadamard(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.LazyLinear(64),
                                nn.LayerNorm(64))
        
        self.l2 = nn.Sequential(nn.LazyLinear(64),
                                nn.LayerNorm(64))

    def forward(self, x, y):
        with torch.no_grad():
            b, n1, n2 = x.shape[0], x.shape[1], y.shape[1]
            x = self.l1(x)
            y = self.l2(y)
            x, y = x.unsqueeze(2), y.unsqueeze(1)
            x, y = x + torch.zeros(b, 1, n2, 64, device=x.device), y + torch.zeros(b, n1, 1, 64, device=x.device)
            one = torch.ones(b, n1, n2, 1, device=x.device)
            xy = x * y
            ret = torch.cat([xy, one], dim=3)
            return ret