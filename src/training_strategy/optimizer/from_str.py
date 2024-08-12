from ..name import optimizer_getfn_name_dict, optimizer_parser_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

def to_optimizer_hyperparameter(optimizer_hyperparameter_str):
    if optimizer_hyperparameter_str[0] == "[":
        optimizer_hyperparameter_str = optimizer_hyperparameter_str[1:-1]
        optimizer_hyperparameter_str = optimizer_hyperparameter_str.split(",")
        optimizer_hyperparameter_str = [to_optimizer_hyperparameter(i) for i in optimizer_hyperparameter_str]
        return optimizer_hyperparameter_str
    
    if optimizer_hyperparameter_str[0].isdigit() is True:
        char_list = [i for i in optimizer_hyperparameter_str]
        try:
            if "." in char_list:
                return float(optimizer_hyperparameter_str)
            else:
                return int(optimizer_hyperparameter_str)
        except:
            return optimizer_hyperparameter_str
    
    if len(optimizer_hyperparameter_str) >= 2:
        if optimizer_hyperparameter_str[1].isdigit() is True and optimizer_hyperparameter_str[0] == "-":
            char_list = [i for i in optimizer_hyperparameter_str]
            try:
                if "." in char_list:
                    return float(optimizer_hyperparameter_str)
                else:
                    return int(optimizer_hyperparameter_str)
            except:
                return optimizer_hyperparameter_str
            
    if optimizer_hyperparameter_str == "True" or optimizer_hyperparameter_str == "true":
        return True
    
    if optimizer_hyperparameter_str == "False" or optimizer_hyperparameter_str == "false":
        return False
    
    if optimizer_hyperparameter_str == "None" or optimizer_hyperparameter_str == "none":
        return None
    
    return optimizer_hyperparameter_str

def parse_optimizer_str(optimizer_str):
    optimizer_str = optimizer_str.split("\n")
    optimizer_str = [i for i in optimizer_str if i != ""]
    optimizer_str = [i.split(" ") for i in optimizer_str]
    settings = {}
    optimizer_name = None
    for i in range(len(optimizer_str)):
        if optimizer_str[i][0].lower() != "name":
            settings[optimizer_str[i][0]] = to_optimizer_hyperparameter(optimizer_str[i][1])
        else:
            optimizer_name = optimizer_str[i][1]
    settings = optimizer_parser_name_dict[optimizer_name.upper()](**settings)
    return settings, optimizer_getfn_name_dict[optimizer_name.upper()]
