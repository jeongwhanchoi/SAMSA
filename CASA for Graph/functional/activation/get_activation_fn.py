import torch
import torch.nn as nn
import torch.nn.functional as F
from functional.activation.approx_gelu import approx_gelu

def get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return approx_gelu
    elif activation == F.gelu:
        return approx_gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")