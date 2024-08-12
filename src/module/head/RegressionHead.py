from ..name import module_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class RegressionHead(nn.Module):
    def __init__(self, 
                 n_classes: int):
        super().__init__()
        self.settings = {
            "name": "REGHEAD",
            "n_classes": n_classes
        }
        self.linear_reg = nn.LazyLinear(n_classes)
        self.linear_att = nn.LazyLinear(1)

    def forward(self, x, mask=None, **kwargs):
        x_p = F.softmax(self.linear_att(x) + (1 - mask.unsqueeze(-1)) * 1e-9, dim=1)
        x = torch.sum(x * x_p, dim=1)
        x = self.linear_reg(x)
        return {'x': x}

module_name_dict["REGHEAD"] = RegressionHead