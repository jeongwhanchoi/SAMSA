import torch
import torch.nn as nn
import torch.nn.functional as F

class pairwise_field(nn.Module):
    def __init__(self):
        super().__init__()
        self.weak_force_limit = nn.Parameter(torch.zeros(1, 1, 1, 1), requires_grad=True)
        self.strong_force_limit = nn.Parameter(torch.zeros(1, 1, 1, 1), requires_grad=True)
        self.scaler_weak = nn.Linear(1, 1, bias=False)
        self.scaler_strong = nn.Linear(1, 1, bias=False)
        self.scaler_gravitation = nn.Linear(1, 1, bias=False)

    def forward(self, x, y):
        with torch.no_grad():
            x_norm = (x**2).sum(2).unsqueeze(2)
            y_t = torch.transpose(y, 1, 2)
            y_norm = (y**2).sum(2).unsqueeze(1)
            dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
            dist = torch.sqrt(F.relu(dist))
            dist = dist.unsqueeze(-1)
        weak = torch.minimum(self.scaler_weak(dist), self.weak_force_limit)
        strong = torch.minimum(self.scaler_strong(dist), self.strong_force_limit)
        gravitation = self.scaler_gravitation(dist)
        return torch.cat([1 / (weak + 1), strong, 1 / (gravitation ** 2 + 1)], dim=-1)