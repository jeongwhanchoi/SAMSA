import torch
import torch.nn as nn
import torch.nn.functional as F
from functional import SAM

class RegressionHead(nn.Module):
    def __init__(self,
                 n_dims: int
                 ):
        super(RegressionHead, self).__init__()
        self.regressor = nn.LazyLinear(n_dims, bias=True)
        self.attention = nn.LazyLinear(1)
        
    def forward(self, x, mask=None):
        if mask is not None:
            x = x * (1 - mask).unsqueeze(-1)
            x_att = F.softmax(self.attention(x) + (mask * (-1e9)).unsqueeze(-1), dim=1)
        else:
            x_att = F.softmax(self.attention(x), dim=1)
        x = torch.sum(x * x_att, dim=1)
        x = self.regressor(x)
        return x