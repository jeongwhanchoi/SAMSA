import torch
import torch.nn as nn

class pairwise_distance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        with torch.no_grad():
            x_norm = (x**2).sum(2).unsqueeze(2)
            y_t = torch.transpose(y, 1, 2)
            y_norm = (y**2).sum(2).unsqueeze(1)
            dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
            dist = dist.unsqueeze(-1)
            return dist