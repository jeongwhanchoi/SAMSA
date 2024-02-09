from layer import RESC
import torch
import torch.nn as nn
import torch.nn.functional as F
# from knn_cuda import KNN
from layer import RESC, RSTL, RandomColor, SoftplusPWP, SAM, GlobalNode, SSCL, LearnableEmbedding, RSTLBlock, RSTLRBlock
# from layer import RESC, LearnableEmbedding, LocalAttention
from head import ClassificationHead, SegmentationHead, RegressionHead

def create_layer(layer_type, hyperparams):
    if layer_type == "LIN":
        return nn.LazyLinear(*hyperparams)
    elif layer_type == "EMB":
        return LearnableEmbedding(*hyperparams)
    elif layer_type == "RESC":
        return RESC(*hyperparams)
    elif layer_type == "GLBN":
        return GlobalNode(*hyperparams)
    elif layer_type == "RSTL":
        return RSTL(*hyperparams)
    elif layer_type == "SSCL":
        return SSCL(*hyperparams)
    elif layer_type == "SPWP":
        return SoftplusPWP(*hyperparams)
    elif layer_type == "CLFH":
        return ClassificationHead(*hyperparams)
    elif layer_type == "SEGH":
        return SegmentationHead(*hyperparams)
    elif layer_type == "NORM":
        return nn.LayerNorm(*hyperparams)
    elif layer_type == "REGH":
        return RegressionHead(*hyperparams)
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


class LRAModel(nn.Module):
    def __init__(self, layer_list):
        super().__init__()
        self.layer_list = nn.ModuleList(layer_list)
        self.is_relative = False
        for i in range(len(self.layer_list)):
            if isinstance(self.layer_list[i], RSTL):
                self.is_relative = True
            elif isinstance(self.layer_list[i], SSCL):
                self.is_relative = True
        if self.is_relative is False:
            self.forward = self.forward_notrelative
        else:
            self.forward = self.forward_relative
        self.layer_len = len(self.layer_list)

    def forward_notrelative(self, x, mask=None):
        for i in range(len(self.layer_list)):
            if isinstance(self.layer_list[i], nn.LazyLinear) or isinstance(self.layer_list[i], nn.Linear):
                x = self.layer_list[i](x)
            elif isinstance(self.layer_list[i], ClassificationHead):
                x, r = self.layer_list[i](x, mask)
            elif isinstance(self.layer_list[i], GlobalNode):
                x, mask = self.layer_list[i](x)
            else:
                x = self.layer_list[i](x, mask)
        return x

    def forward_relative(self, x, mask=None): # relative
        x_ref = torch.stack([torch.arange(x.shape[1]) for _ in range(x.shape[0])], dim=0).reshape(x.shape[0], 1, x.shape[1]).transpose(1, 2)
        x_ref = x_ref.to(x.device)
        x_ref = torch.cat([x_ref, torch.ones(x_ref.shape[0], x_ref.shape[1], 1, device=x_ref.device)], dim=2)

        #Random features
        x = torch.cat([x, torch.randn_like(x)], dim=-1)
        
        for i in range(len(self.layer_list)):
            if isinstance(self.layer_list[i], nn.LazyLinear) or isinstance(self.layer_list[i], nn.Linear):
                x = self.layer_list[i](x)
            elif isinstance(self.layer_list[i], ClassificationHead):
                x, r = self.layer_list[i](x, mask)
            elif isinstance(self.layer_list[i], RSTL) or isinstance(self.layer_list[i], RSTLBlock) or isinstance(self.layer_list[i], RSTLRBlock):
                x, r = self.layer_list[i](x, x_ref, mask)
            elif isinstance(self.layer_list[i], SSCL):
                x = self.layer_list[i](x, x_ref, mask)
            elif isinstance(self.layer_list[i], GlobalNode):
                x, x_ref, mask = self.layer_list[i](x, x_ref, mask)
            else:
                x = self.layer_list[i](x, mask)
        return x

    def __len__(self):
        return len(self.layer_list)

    def __getitem__(self, idx):
        return self.layer_list[idx]
    
def parse_lra_architecture(arch_string):
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
    return LRAModel(sequential_layer)