from ..name import module_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationHead(nn.Module):
    def __init__(self, 
                 n_classes: int):
        super().__init__()
        self.settings = {
            "name": "SEGHEAD",
            "n_classes": n_classes
        }
        self.linear = nn.LazyLinear(n_classes)

    def forward(self, x):
        return {'x': self.linear(x)}

module_name_dict["SEGHEAD"] = SegmentationHead