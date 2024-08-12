from ..name import module_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class Coords(nn.Module):
    def forward(self, x, **kwargs):
        return {'x_coords': x[:, :, :3]}

class Save(nn.Module):
    def __init__(self, name='x_saved'):
        super().__init__()
        self.name = name

    def forward(self, x, **kwargs):
        return {self.name: x}

class Merge(nn.Module):
    def __init__(self, name='x_saved'):
        super().__init__()
        self.name = name
        self.scaler = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, **kwargs):
        return {'x': x * self.scaler + kwargs[self.name]}

module_name_dict['COORDS'] = Coords
module_name_dict['SAVE'] = Save
module_name_dict['MERGE'] = Merge