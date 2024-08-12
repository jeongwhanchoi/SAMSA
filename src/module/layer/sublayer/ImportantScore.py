import torch
import torch.nn as nn
import torch.nn.functional as F

from .RMSNorm import RMSNorm

import math

class ImportantScore(nn.Module):
    def __init__(self,
                 nhead: int,
                 n_dimension: int):
        super().__init__()
        self.importance_score = nn.Sequential(
            nn.Linear(n_dimension, n_dimension),
            nn.GELU(),
            nn.Linear(n_dimension, n_dimension)
        )

        self.norm = RMSNorm(n_dimension)

        self.transf = nn.Linear(n_dimension, nhead)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor):
        b, n = x.shape[0], x.shape[1]
        ret = self.transf(x + self.norm(self.importance_score(x))) + ((1 - mask) * (-1e9)).reshape(b, n, 1)
        return ret