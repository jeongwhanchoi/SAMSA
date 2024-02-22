import torch

def identity(x):
    with torch.no_grad():
        return x.unsqueeze(-1)