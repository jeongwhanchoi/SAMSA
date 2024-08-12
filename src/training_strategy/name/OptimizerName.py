import torch
import torch.nn as nn

def init_name_optimizer_dict():
    if not ('optimizer_parser_name_dict' in globals()):
        global optimizer_parser_name_dict
        optimizer_parser_name_dict = {}

    if not ('optimizer_getfn_name_dict' in globals()):
        global optimizer_getfn_name_dict
        optimizer_getfn_name_dict = {}