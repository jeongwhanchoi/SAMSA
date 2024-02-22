import torch.nn.functional as F

def approx_gelu(x):
    return F.gelu(x, approximate='tanh')