import torch
import torch.nn as nn

def init_name_dataset_dict():
    if not ('dataset_parser_name_dict' in globals()):
        global dataset_parser_name_dict
        dataset_parser_name_dict = {}