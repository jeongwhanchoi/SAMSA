import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import TransformerEncoderBlock, RandomColor, ClassificationHead, SegmentationHead

def create_layer(layer_type, hyperparams):
    if layer_type == "LIN":
        return nn.LazyLinear(*hyperparams)
    if layer_type == "RC":
        return RandomColor()
    elif layer_type == "TRAN":
        return TransformerEncoderBlock(*hyperparams)
    elif layer_type == "CLFH":
        return ClassificationHead(*hyperparams)
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

class PointModel(nn.Module):
    def __init__(self, layer_list, is_normal=False, input_mask=True):
        super().__init__()
        self.layer_list = nn.ModuleList(layer_list)
        self.layer_len = len(self.layer_list)
        self.is_normal = is_normal
        self.input_mask = input_mask
        self.normalization = nn.BatchNorm1d(1, affine=False)

    def forward(self, x, mask=None): # relative
        x_ref_ = x[:, :, :3] * 1
        x_ref_ = self.normalization(x_ref_.reshape(-1, 1)).reshape(x_ref_.shape)
        x[:,:,:3] = x_ref_
        if self.is_normal == False:
            x_ref = x[:, :, :3] * 1
            if self.input_mask is True:
                x[:, :, :3] = 1.0
        else:
            x_ref = x[:, :, :6] * 1
            if self.input_mask is True:
                x[:, :, :6] = 1.0

        #Random features
        x = torch.cat([x, torch.randn(x.shape[0], x.shape[1], 256, device=x.device)], dim=-1)
        
        for i in range(len(self.layer_list)):
            if isinstance(self.layer_list[i], nn.LazyLinear) or isinstance(self.layer_list[i], nn.Linear):
                x = self.layer_list[i](x)
            elif isinstance(self.layer_list[i], TransformerEncoderBlock):
                x = self.layer_list[i](x, x_ref, mask)
            else:
                x = self.layer_list[i](x, mask)
        return x

    def __len__(self):
        return len(self.layer_list)

    def __getitem__(self, idx):
        return self.layer_list[idx]

def parse_point_architecture(arch_string, is_normal=False, input_mask=True):
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
    return PointModel(sequential_layer, is_normal, input_mask)