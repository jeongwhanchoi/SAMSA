import torch
import torch.nn as nn
import torch.nn.functional as F

def to_settings_hyperparameter(settings_hyperparameter_str):
    if settings_hyperparameter_str[0] == "[":
        settings_hyperparameter_str = settings_hyperparameter_str[1:-1]
        settings_hyperparameter_str = settings_hyperparameter_str.split(",")
        settings_hyperparameter_str = [to_settings_hyperparameter(i) for i in settings_hyperparameter_str]
        return settings_hyperparameter_str
    
    if settings_hyperparameter_str[0].isdigit() is True:
        char_list = [i for i in settings_hyperparameter_str]
        try:
            if "." in char_list:
                return float(settings_hyperparameter_str)
            else:
                return int(settings_hyperparameter_str)
        except:
            return settings_hyperparameter_str
    
    if len(settings_hyperparameter_str) >= 2:
        if settings_hyperparameter_str[1].isdigit() is True and settings_hyperparameter_str[0] == "-":
            char_list = [i for i in settings_hyperparameter_str]
            try:
                if "." in char_list:
                    return float(settings_hyperparameter_str)
                else:
                    return int(settings_hyperparameter_str)
            except:
                return settings_hyperparameter_str
            
    if settings_hyperparameter_str == "True" or settings_hyperparameter_str == "true":
        return True
    
    if settings_hyperparameter_str == "False" or settings_hyperparameter_str == "false":
        return False
    
    if settings_hyperparameter_str == "None" or settings_hyperparameter_str == "none":
        return None
    
    return settings_hyperparameter_str

def parse_settings_str(settings_str):
    settings_str = settings_str.split("\n")
    settings_str = [i for i in settings_str if i != ""]
    settings_str = [i.split(" ") for i in settings_str]
    settings = {}
    for i in range(len(settings_str)):
        settings[settings_str[i][0]] = to_settings_hyperparameter(settings_str[i][1])
    return settings
