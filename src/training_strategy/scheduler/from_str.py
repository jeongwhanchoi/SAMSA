from ..name import scheduler_getfn_name_dict, scheduler_parser_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

def to_scheduler_hyperparameter(scheduler_hyperparameter_str):
    if scheduler_hyperparameter_str[0] == "[":
        scheduler_hyperparameter_str = scheduler_hyperparameter_str[1:-1]
        scheduler_hyperparameter_str = scheduler_hyperparameter_str.split(",")
        scheduler_hyperparameter_str = [to_scheduler_hyperparameter(i) for i in scheduler_hyperparameter_str]
        return scheduler_hyperparameter_str
    
    if scheduler_hyperparameter_str[0].isdigit() is True:
        char_list = [i for i in scheduler_hyperparameter_str]
        try:
            if "." in char_list:
                return float(scheduler_hyperparameter_str)
            else:
                return int(scheduler_hyperparameter_str)
        except:
            return scheduler_hyperparameter_str
    
    if len(scheduler_hyperparameter_str) >= 2:
        if scheduler_hyperparameter_str[1].isdigit() is True and scheduler_hyperparameter_str[0] == "-":
            char_list = [i for i in scheduler_hyperparameter_str]
            try:
                if "." in char_list:
                    return float(scheduler_hyperparameter_str)
                else:
                    return int(scheduler_hyperparameter_str)
            except:
                return scheduler_hyperparameter_str
            
    if scheduler_hyperparameter_str == "True" or scheduler_hyperparameter_str == "true":
        return True
    
    if scheduler_hyperparameter_str == "False" or scheduler_hyperparameter_str == "false":
        return False
    
    if scheduler_hyperparameter_str == "None" or scheduler_hyperparameter_str == "none":
        return None
    
    return scheduler_hyperparameter_str

def parse_scheduler_str(scheduler_str):
    scheduler_str = scheduler_str.split("\n")
    scheduler_str = [i for i in scheduler_str if i != ""]
    scheduler_str = [i.split(" ") for i in scheduler_str]
    settings = {}
    scheduler_name = None
    for i in range(len(scheduler_str)):
        if scheduler_str[i][0].lower() != "name":
            settings[scheduler_str[i][0]] = to_scheduler_hyperparameter(scheduler_str[i][1])
        else:
            scheduler_name = scheduler_str[i][1]
    settings = scheduler_parser_name_dict[scheduler_name.upper()](**settings)
    return settings, scheduler_getfn_name_dict[scheduler_name.upper()]
