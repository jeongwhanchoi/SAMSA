from .name import module_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

def to_arch_hyperparameter(arch_hyperparameter_str):
    if arch_hyperparameter_str[0].isdigit() is True:
        char_list = [i for i in arch_hyperparameter_str]
        try:
            if "." in char_list:
                return float(arch_hyperparameter_str)
            else:
                return int(arch_hyperparameter_str)
        except:
            return arch_hyperparameter_str
    
    if len(arch_hyperparameter_str) >= 2:
        if arch_hyperparameter_str[1].isdigit() is True and arch_hyperparameter_str[0] == "-":
            char_list = [i for i in arch_hyperparameter_str]
            try:
                if "." in char_list:
                    return float(arch_hyperparameter_str)
                else:
                    return int(arch_hyperparameter_str)
            except:
                return arch_hyperparameter_str
            
    if arch_hyperparameter_str == "True" or arch_hyperparameter_str == "true":
        return True
    
    if arch_hyperparameter_str == "False" or arch_hyperparameter_str == "false":
        return False
    
    if arch_hyperparameter_str == "None" or arch_hyperparameter_str == "none":
        return None

    return arch_hyperparameter_str

def convert_arch_str_list(lst):
    lst_ = []
    for i in range(len(lst)):
        try:
            lst_.append(to_arch_hyperparameter(lst[i]))
        except:
            pass
    return lst_

class Seq(nn.Module):
    def __init__(self, mdl):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        if len(mdl) != 0:
            self.module_list = nn.ModuleList(mdl)
        else:
            self.module_list = nn.ModuleList()

    def append(self, layer):
        self.module_list.append(layer)

    def update(self, variable_dict, output_dict):
        if isinstance(output_dict, torch.Tensor):
            variable_dict['x'] = output_dict
            return variable_dict
        for key, value in output_dict.items():
            variable_dict[key] = value
        return variable_dict

    def forward(self, **kwargs):
        variable_dict = kwargs
        variable_dict['residual_scale'] = self.residual_scale
        for i in range(len(self.module_list)):
            self.update(variable_dict, self.module_list[i](**variable_dict))
        return variable_dict['x']

def str_to_sequential_model(model_str):
    model = Seq([])
    model_str = model_str.split('\n')
    model_str = [i.split(',') for i in model_str]
    model_str = [convert_arch_str_list(i) for i in model_str]
    for i in range(len(model_str)):
        model.append(module_name_dict[model_str[i][0]](*(model_str[i][1:])))
    return model

def str_to_module_list(model_str):
    model = nn.ModuleList()
    model_str = model_str.split('\n')
    model_str = [i.split(',') for i in model_str]
    model_str = [convert_arch_str_list(i) for i in model_str]
    for i in range(len(model_str)):
        model.append(module_name_dict[model_str[i][0]](*(model_str[i][1:])))
    return model