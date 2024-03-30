import torch
import torch.nn as nn
import math

class pairwise_similarity(nn.Module):
    def __init__(self, nhead):
        super().__init__()
        self.nhead = nhead
        self.l1 = nn.Sequential(nn.LazyLinear(512),
                                nn.LayerNorm(512),
                                nn.GELU(),
                                nn.LazyLinear(512))
        
        self.l2 = nn.Sequential(nn.LazyLinear(512),
                                nn.LayerNorm(512),
                                nn.GELU(),
                                nn.LazyLinear(512))

    def forward(self, x, y):
        with torch.no_grad():
            b, n1, n2 = x.shape[0], x.shape[1], y.shape[1]
            x = self.l1(x)
            y = self.l2(y)
            x = x.reshape(b, n1, self.nhead, 512 // self.nhead).transpose(1, 2)
            y = y.reshape(b, n2, self.nhead, 512 // self.nhead).transpose(1, 2)
            smap = x @ y.transpose(2, 3) / math.sqrt(512 // self.nhead)
            smap = smap.permute(0, 2, 3, 1)
            one = torch.ones(b, n1, n2, 1, device=x.device)
            smap = torch.cat([smap, one], dim=3)
            return smap