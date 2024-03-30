import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import TransformerEncoderBlock, ClassificationHead, SegmentationHead, RegressionHead
import math

def create_layer(layer_type, hyperparams):
    if layer_type == "LIN":
        return nn.LazyLinear(*hyperparams)
    elif layer_type == "TRAN":
        return TransformerEncoderBlock(*hyperparams)
    elif layer_type == "CLFH":
        return ClassificationHead(*hyperparams)
    elif layer_type == "REGH":
        return RegressionHead(*hyperparams)
    elif layer_type == "SEGH":
        return SegmentationHead(*hyperparams)
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")

def to_number(x):
    isfloat = False
    for i in range(len(x)):
        if x[i] == '.':
            isfloat = True
    if isfloat == True:
        x = float(x)
    else:
        x = int(x)
    return x

def to_var(x):
    if x[0].isdigit():
        return to_number(x)
    if x in ["True", "False"]:
        return x == "True"
    return x # String

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class SequenceModel(nn.Module):
    def __init__(self, layer_list, seq_length):
        super().__init__()
        self.layer_list = nn.ModuleList(layer_list)
        self.layer_len = len(self.layer_list)

        self.lpe = nn.Parameter(positionalencoding1d(2048, seq_length).unsqueeze(0), requires_grad=False)

        # pes = [torch.randn(1, 1, 64)]
        # dampen = [(1 - i / 512) for i in range(32, 96)]
        # dampen = torch.tensor(dampen).reshape(1, 1, 64)
        # for _ in range(4095):
        #     pes.append(pes[-1] * dampen + torch.randn(1, 1, 64) * (1 - dampen))
        # self.lpe = nn.Parameter(torch.cat(pes, dim=1), requires_grad=True)

        # for i in range(self.layer_len):
        #     if isinstance(self.layer_list[i], TransformerEncoderBlock):
        #         self.lpe.append(nn.Parameter(torch.randn(1, 4096, 512), requires_grad=True))
        #     else:
        #         self.lpe.append(nn.Parameter(torch.randn(1), requires_grad=False))
        # self.lpe = nn.ParameterList(self.lpe)

        # self.lpe = nn.Parameter(torch.randn(1, 4096, 512) * 0.001, requires_grad=True)

    def forward(self, x, mask=None): # relative
        # #Random features
        # x = torch.cat([x, torch.randn(x.shape[0], x.shape[1], 256, device=x.device)], dim=-1)
        
        for i in range(len(self.layer_list)):
            if isinstance(self.layer_list[i], nn.LazyLinear) or isinstance(self.layer_list[i], nn.Linear):
                x = self.layer_list[i](x)
            elif isinstance(self.layer_list[i], TransformerEncoderBlock):
                x = self.layer_list[i](x, self.lpe[:,:x.shape[1],:].repeat(x.shape[0], 1, 1), None, mask)
            else:
                x = self.layer_list[i](x, mask)
        return x

    def __len__(self):
        return len(self.layer_list)

    def __getitem__(self, idx):
        return self.layer_list[idx]

def parse_sequence_architecture(arch_string, seqlength):
    layer = arch_string.split('@')
    sequential_layer = []
    
    for layer_str in layer:
        layer_type, *hyperparams = layer_str.split(',')
        
        # Convert hyperparameters to the appropriate types
        hyperparams = [to_var(param) for param in hyperparams]
        
        # Convert to the desired format
        layer_str = f"{layer_type}({', '.join(map(str, hyperparams))})"
        
        # Wrap each hyperparameter individually with parentheses
        for i in range(len(hyperparams)):
            layer_str = layer_str.replace(str(hyperparams[i]), f"({hyperparams[i]})", 1)
        layer = create_layer(layer_type, hyperparams)
        sequential_layer.append(layer)
    
    # print("Model architecture consists of: \n", sequential_layer)
    return torch.compile(SequenceModel(sequential_layer, seqlength))