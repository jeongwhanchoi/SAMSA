import torch
import torch.nn as nn

class pairwise_concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        with torch.no_grad():
            # x(b, n1, d1), y(b, n2, d2)
            b, n1, d1, n2, d2 = x.shape[0], x.shape[1], x.shape[2], y.shape[1], y.shape[2]
            x, y = x.unsqueeze(2), y.unsqueeze(1)
            x, y = x + torch.zeros(b, 1, n2, d1, device=x.device), y + torch.zeros(b, n1, 1, d2, device=x.device)
            one = torch.ones(b, n1, n2, 1, device=x.device)
            ret = torch.cat([x, y, one], dim=3)
            return ret