import torch
import torch.nn as nn
from layer import SAM
    
class ClassificationHead(nn.Module):
    def __init__(self,
                 n_classes: int,
                 d_model: int,
                 n_sampled_points_lb: int = None,
                 n_sampled_points_ub: int = None,
                 drop_point: float = 0.1,
                 ):
        super(ClassificationHead, self).__init__()
        
        self.forw = SAM(n_sampled_points_lb,
                        n_sampled_points_ub,
                        d_model,
                        drop_point)
        
        self.classifier = nn.LazyLinear(n_classes, bias=True)
        
    def forward(self, x, mask=None):
        x, regularization = self.forw(x, mask=mask)
        x, _ = torch.max(x, dim=1)
        x = self.classifier(x)
        return x, regularization
