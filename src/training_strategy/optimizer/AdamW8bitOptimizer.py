from ..name import optimizer_getfn_name_dict, optimizer_parser_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import bitsandbytes.optim as optim


def parse_adamw8bit_optimizer(lr=0.001,
                              betas=[0.9,0.999],
                          eps=1e-08,
                          weight_decay=0,
                          amsgrad=False,
                          ):
    return {
        "lr" : lr,
        "betas" : betas,
        "eps" : eps,
        "weight_decay" : weight_decay,
        "amsgrad" : amsgrad,
    }

def get_adamw8bit_optimizer(params,
                       setting_dict):
    setting_dict['params'] = params
    return optim.AdamW8bit(**setting_dict)

optimizer_parser_name_dict['ADAMW8'] = parse_adamw8bit_optimizer
optimizer_getfn_name_dict['ADAMW8'] = get_adamw8bit_optimizer