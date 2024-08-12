from ..name import optimizer_getfn_name_dict, optimizer_parser_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import lpmm.optim as optim

def parse_lbadamw_optimizer(lr=0.001,
                          betas=[0.9,0.999],
                          eps=1e-08,
                          weight_decay=0,
                          fused=False):
    return {
        "lr" : lr,
        "betas" : betas,
        "eps" : eps,
        "weight_decay" : weight_decay,
        "fused" : fused
    }

def get_lbadamw_optimizer(params,
                       setting_dict):
    setting_dict['params'] = params
    return optim.AdamW(**setting_dict)

optimizer_parser_name_dict['LBADAMW'] = parse_lbadamw_optimizer
optimizer_getfn_name_dict['LBADAMW'] = get_lbadamw_optimizer