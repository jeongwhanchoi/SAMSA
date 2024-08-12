from ..name import scheduler_getfn_name_dict, scheduler_parser_name_dict

import torch.optim.lr_scheduler as lr_scheduler

def parse_cosine_scheduler(T_max):
    return {"T_max": T_max}

def get_cosine_scheduler(optimizer, setting_dict):
    return lr_scheduler.CosineAnnealingLR(optimizer, **setting_dict)

scheduler_parser_name_dict['COSINE'] = parse_cosine_scheduler
scheduler_getfn_name_dict['COSINE'] = get_cosine_scheduler