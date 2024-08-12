from ..name import optimizer_getfn_name_dict, optimizer_parser_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

def parse_sgd_optimizer(lr=0.001,
                        momentum=0.9,
                        dampening=0.999,
                        weight_decay=0,
                        nesterov=False,
                        maximize=False,
                        differentiable=False,
                        fused=None):
    return {
        "lr" : lr,
        "momentum" : momentum,
        "dampening" : dampening,
        "weight_decay" : weight_decay,
        "nesterov" : nesterov,
        "maximize" : maximize,
        "differentiable" : differentiable,
        "fused" : fused
    }

def get_sgd_optimizer(params,
                      setting_dict):
    setting_dict['params'] = params
    return optim.SGD(**setting_dict)

optimizer_parser_name_dict['SGD'] = parse_sgd_optimizer
optimizer_getfn_name_dict['SGD'] = get_sgd_optimizer