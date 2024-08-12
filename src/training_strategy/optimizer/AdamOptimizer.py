from ..name import optimizer_getfn_name_dict, optimizer_parser_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

def parse_adam_optimizer(lr=0.001,
                         betas=[0.9, 0.999],
                         eps=1e-08,
                         weight_decay=0,
                         amsgrad=False,
                         maximize=False,
                         differentiable=False):
    return {
        "lr" : lr,
        "betas" : betas,
        "eps" : eps,
        "weight_decay" : weight_decay,
        "amsgrad" : amsgrad,
        "maximize" : maximize,
        "differentiable" : differentiable,
    }

def get_adam_optimizer(params,
                       setting_dict):
    setting_dict['params'] = params
    return optim.Adam(**setting_dict)

optimizer_parser_name_dict['ADAM'] = parse_adam_optimizer
optimizer_getfn_name_dict['ADAM'] = get_adam_optimizer