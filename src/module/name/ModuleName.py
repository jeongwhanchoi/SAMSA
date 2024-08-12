import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.l = nn.Linear(*args)
    
    def forward(self, x, **args):
        return {'x': self.l(x)}

class LazyLinear(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.l = nn.LazyLinear(*args)
    
    def forward(self, x, **args):
        return {'x': self.l(x)}

def init_name_method_dict():
    if not ('module_name_dict' in globals()):
        global module_name_dict
        module_name_dict = {
            "LINEAR": Linear,
            "LAZYLINEAR": LazyLinear,
        }