from ..name import dataset_parser_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

def to_dataset_hyperparameter(dataset_hyperparameter_str):
    if dataset_hyperparameter_str[0] == "[":
        dataset_hyperparameter_str = dataset_hyperparameter_str[1:-1]
        dataset_hyperparameter_str = dataset_hyperparameter_str.split(",")
        dataset_hyperparameter_str = [to_dataset_hyperparameter(i) for i in dataset_hyperparameter_str]
        return dataset_hyperparameter_str
    
    if dataset_hyperparameter_str[0].isdigit() is True:
        char_list = [i for i in dataset_hyperparameter_str]
        try:
            if "." in char_list:
                return float(dataset_hyperparameter_str)
            else:
                return int(dataset_hyperparameter_str)
        except:
            return dataset_hyperparameter_str
    
    if len(dataset_hyperparameter_str) >= 2:
        if dataset_hyperparameter_str[1].isdigit() is True and dataset_hyperparameter_str[0] == "-":
            char_list = [i for i in dataset_hyperparameter_str]
            try:
                if "." in char_list:
                    return float(dataset_hyperparameter_str)
                else:
                    return int(dataset_hyperparameter_str)
            except:
                return dataset_hyperparameter_str
            
    if dataset_hyperparameter_str == "True" or dataset_hyperparameter_str == "true":
        return True
    
    if dataset_hyperparameter_str == "False" or dataset_hyperparameter_str == "false":
        return False
    
    if dataset_hyperparameter_str == "None" or dataset_hyperparameter_str == "none":
        return None
    
    return dataset_hyperparameter_str

def parse_dataset_str(dataset_str):
    dataset_str = dataset_str.split("\n")
    dataset_str = [i for i in dataset_str if i != ""]
    dataset_str = [i.split(" ") for i in dataset_str]
    settings = {}
    dataset_name = None
    for i in range(len(dataset_str)):
        if dataset_str[i][0].lower() != "name":
            settings[dataset_str[i][0]] = to_dataset_hyperparameter(dataset_str[i][1])
        else:
            dataset_name = dataset_str[i][1]
    return dataset_parser_name_dict[dataset_name.upper()], settings
