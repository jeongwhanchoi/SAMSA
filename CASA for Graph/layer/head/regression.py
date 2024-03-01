import torch
import torch.nn as nn
from functional import SAM

class RegressionHead(nn.Module):
    def __init__(self,
                 n_dims: int,
                 d_model: int,
                 n_sampled_points: int = None,
                 drop_point: float = 0.1,
                 ):
        super(RegressionHead, self).__init__()
        
        self.forw = SAM(n_sampled_points,
                        d_model,
                        drop_point)
        
        self.regressor = nn.LazyLinear(n_dims, bias=True)
        
    def forward(self, x, mask=None):
        x, _, _, _ = self.forw(x, mask=mask)
        x = torch.mean(x, dim=1)
        x = self.regressor(x)
        return x