from ..name import scheduler_getfn_name_dict, scheduler_parser_name_dict

import torch.optim.lr_scheduler as lr_scheduler

def parse_step_lr_scheduler(step_size=30, gamma=0.1):
    return {
        "step_size": step_size,
        "gamma": gamma
    }

def get_step_lr_scheduler(optimizer, setting_dict):
    return lr_scheduler.StepLR(optimizer, **setting_dict)

scheduler_parser_name_dict['STEPLR'] = parse_step_lr_scheduler
scheduler_getfn_name_dict['STEPLR'] = get_step_lr_scheduler