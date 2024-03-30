import torch
import torch.nn as nn

class pairwise_placeholder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        with torch.no_grad():
            return torch.zeros(x.shape[0], x.shape[1], y.shape[1], 1, device=x.device)