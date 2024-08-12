from ..name import metrics_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

def to_lossfn_hyperparameter(lossfn_hyperparameter_str):
    if lossfn_hyperparameter_str[0].isdigit() is True:
        char_list = [i for i in lossfn_hyperparameter_str]
        try:
            if "." in char_list:
                return float(lossfn_hyperparameter_str)
            else:
                return int(lossfn_hyperparameter_str)
        except:
            return lossfn_hyperparameter_str
    
    if len(lossfn_hyperparameter_str) >= 2:
        if lossfn_hyperparameter_str[1].isdigit() is True and lossfn_hyperparameter_str[0] == "-":
            char_list = [i for i in lossfn_hyperparameter_str]
            try:
                if "." in char_list:
                    return float(lossfn_hyperparameter_str)
                else:
                    return int(lossfn_hyperparameter_str)
            except:
                return lossfn_hyperparameter_str
            
    if lossfn_hyperparameter_str == "True" or lossfn_hyperparameter_str == "true":
        return True
    
    if lossfn_hyperparameter_str == "False" or lossfn_hyperparameter_str == "false":
        return False
    
    if lossfn_hyperparameter_str == "None" or lossfn_hyperparameter_str == "none":
        return None
    
    return lossfn_hyperparameter_str

def convert_lossfn_str_list(lst):
    lst_ = []
    for i in range(len(lst)):
        try:
            lst_.append(to_lossfn_hyperparameter(lst[i]))
        except:
            pass
    return lst_

class MeanLossWrapper(nn.Module):
    def __init__(self, list_of_loss_fn):
        super().__init__()
        self.list_of_loss_fn = list_of_loss_fn
        self.len = len(self.list_of_loss_fn)
        self.settings = {
            "name": "",
        }
        for i in range(len(list_of_loss_fn)):
            self.settings["name"] += list_of_loss_fn[i].__class__.__name__ + "&"
        self.settings["name"] = self.settings["name"][:-1]

    def forward(self, *args, **kwargs):
        loss = self.list_of_loss_fn[0](*args, **kwargs)
        for i in range(1, self.len):
            loss += self.list_of_loss_fn[i](*args, **kwargs)
        return loss / self.len
    
    def __len__(self):
        return self.len

def parse_lossfunction_str(metrics_str):
    metrics = nn.ModuleList()
    metrics_str = metrics_str.split('\n')
    metrics_str = [i.split(',') for i in metrics_str]
    metrics_str = [convert_lossfn_str_list(i) for i in metrics_str]
    for i in range(len(metrics_str)):
        metrics.append(metrics_name_dict[metrics_str[i][0]](*(metrics_str[i][1:])))
    return MeanLossWrapper(metrics)