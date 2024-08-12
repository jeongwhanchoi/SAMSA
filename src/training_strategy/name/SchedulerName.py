import torch
import torch.nn as nn

def init_name_scheduler_dict():
    if not ('scheduler_parser_name_dict' in globals()):
        global scheduler_parser_name_dict
        scheduler_parser_name_dict = {}

    if not ('scheduler_getfn_name_dict' in globals()):
        global scheduler_getfn_name_dict
        scheduler_getfn_name_dict = {}