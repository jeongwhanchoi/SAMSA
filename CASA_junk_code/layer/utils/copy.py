import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch import Tensor
from typing import List, Union, Any, Callable, Optional

class Copy(Function):
    @staticmethod
    def forward(ctx, input, k):
        ctx.k = k
        output = torch.cat([input for _ in range(k)], dim=2)
        b, n, d = output.shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        b, n, d = grad_output.shape
        grad_output = grad_output.reshape(b, n, ctx.k, d // ctx.k)
        return torch.mean(grad_output, dim=2), None