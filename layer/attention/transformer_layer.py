import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layer.attention.mha import MultiHeadAttention
from functional import get_activation_fn
from typing import List, Union, Any, Callable, Optional

class TransformerEncoderLayer(nn.Module):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int,
                 n_sampled_points, drop_point,
                 score_functions, relative_function, probability_function,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, 
                                            nhead, 
                                            n_sampled_points, 
                                            drop_point, 
                                            score_functions, 
                                            relative_function, 
                                            probability_function)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.res_weight = nn.Parameter(torch.zeros(1), requires_grad=True)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(
            self,
            src: Tensor,
            position,
            mask=None) -> Tensor:

        x = src
        if self.norm_first:
            x_ = self._sa_block(self.norm1(x), position, mask)
            x = x[:,:x_.shape[1],:] + x_
            x = x + self._ff_block(self.norm2(x))
        else:
            x_ = self._sa_block(x, position, mask)
            x = self.norm1(x[:,:x_.shape[1],:] + x_)
            x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor, position, mask=None) -> Tensor:
        x, _ = self.self_attn(x, x, x, position, mask)
        x = x * self.res_weight
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x)))) * self.res_weight
        return self.dropout2(x)