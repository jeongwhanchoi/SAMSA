import torch
import torch.nn as nn
from functional import SAM
    
class RegressionHead(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_sampled_points_lb: int = None,
                 n_sampled_points_ub: int = None,
                 n_center: int = None,
                 drop_point: float = 0.1,
                 ):
        super(RegressionHead, self).__init__()
        
        self.forw = SAM(n_sampled_points_lb,
                        n_sampled_points_ub,
                        n_center,
                        d_model,
                        drop_point)
        
        self.regression = nn.LazyLinear(d_model, bias=True)
        
    def forward(self, x, mask=None):
        x, _ = self.forw(x, mask)
        x, _ = torch.mean(x, dim=1)
        x = self.regression(x)
        return x
