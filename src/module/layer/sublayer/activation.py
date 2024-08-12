from ...name import module_name_dict

import torch.nn as nn
import torch.nn.functional as F

class GeGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.settings = dict(locals())
        self.settings["name"] = "geglu"
        for k, v in self.settings.items():
            self.settings[k] = str(v)

    def forward(self, x, **args):
        x, gates = x.chunk(2, dim = -1)
        x = x * F.gelu(gates)
        return {'x': x}
    
class SqRELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, **args):
        if self.training is True:
            surrogate = 2 * F.relu(x) - 1
            return {'x': (F.relu(x) ** 2 - surrogate).detach() + surrogate}
        else:
            return {'x': F.relu(x) ** 2}
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, **args):
        return {'x': F.gelu(x, approximate='tanh')}
    
module_name_dict["gelu"] = GELU
module_name_dict["geglu"] = GeGLU
module_name_dict["sqrelu"] = SqRELU

__all__ = ['GELU',
           'GeGLU',
           'SqRELU']