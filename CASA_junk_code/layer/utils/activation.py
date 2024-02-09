import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch import Tensor
from typing import List, Union, Any, Callable, Optional

def approx_gelu(x):
    return F.gelu(x, approximate='tanh')

class LearnableReverseReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        return F.softplus(x * -1) * -1 + F.softplus(self.b)
    
class dropRelu(nn.Module):
    def __init__(self, keep_prob=0.95):
        '''
        :param keep_prob: the probability of retaining the ReLU activation
        '''
        super(dropRelu, self).__init__()
        self.keep_prob = keep_prob
    def forward(self, x):
        '''
        :param x: input of x
        :return: drop activation during training or testing phase
        '''
        size_len = len(x.size())
        if self.training:
            Bernoulli_mask = torch.cuda.FloatTensor(x.size()[0:size_len]).fill_(1)
            Bernoulli_mask.bernoulli_(self.keep_prob)
            temp = torch.Tensor().cuda()
            output = torch.Tensor().cuda()
            temp.resize_as_(x).copy_(x)
            output.resize_as_(x).copy_(x)
            output.mul_(Bernoulli_mask)
            output.mul_(-1)
            output.add_(temp)
            temp.clamp_(min = 0)
            temp.mul_(Bernoulli_mask)
            output.add_(temp)
            return output
        else:
            temp = torch.Tensor().cuda()
            output = torch.Tensor().cuda()
            temp.resize_as_(x).copy_(x)
            output.resize_as_(x).copy_(x)
            output.mul_(self.keep_prob)
            output.mul_(-1)
            output.add_(temp)
            temp.clamp_(min=0)
            temp.mul_(self.keep_prob)
            output.add_(temp)
            return output

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return approx_gelu
    elif activation == F.gelu:
        return approx_gelu
    if activation == "drelu":
        return dropRelu()

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")