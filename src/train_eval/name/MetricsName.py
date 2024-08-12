import torch
import torch.nn as nn

def init_name_metrics_dict():
    if not ('metrics_name_dict' in globals()):
        global metrics_name_dict
        metrics_name_dict = {
            "CRENP": nn.CrossEntropyLoss,
            "MAE": nn.L1Loss,
            "MSE": nn.MSELoss
        }