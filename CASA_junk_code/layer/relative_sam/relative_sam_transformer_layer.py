import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layer.utils import GRSoftmax, euclidean_score, scale_dot_score, composite_score
from layer.leaky_attention.mha import RelativeScaledDotSAMMultiHeadAttention, RelativeEuclideanSAMMultiHeadAttention, RelativeCompositeSAMMultiHeadAttention, RelativeNeuralSAMMultiHeadAttention
from layer.norm import SequenceFeatureNorm
from typing import List, Union, Any, Callable, Optional
from layer.utils import _get_activation_fn

class RelativeCustomSAMTransformerEncoderLayer(nn.Module):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int,
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None, attention_type=RelativeScaledDotSAMMultiHeadAttention,
                 normtype="LayerNorm") -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = attention_type(d_model, nhead, n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points=drop_global_points)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        if normtype == "LayerNorm":
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        elif normtype == "FeatureNorm":
            self.norm1 = SequenceFeatureNorm(d_model)
            self.norm2 = SequenceFeatureNorm(d_model)
        else:
            raise ValueError
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.res_weight = nn.Parameter(torch.zeros(1), requires_grad=True)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

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
            x_, regularization = self._sa_block(self.norm1(x), position, mask)
            x = x[:,:x_.shape[1],:] + x_
            x = x + self._ff_block(self.norm2(x))
        else:
            x_, regularization = self._sa_block(x, position, mask)
            x = self.norm1(x[:,:x_.shape[1],:] + x_)
            x = self.norm2(x + self._ff_block(x))
        return x, regularization

    # self-attention block
    def _sa_block(self, x: Tensor, position, mask=None) -> Tensor:
        x, _, regularization = self.self_attn(x, x, x, position, mask)
        x = x * self.res_weight
        return self.dropout1(x), regularization

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x)))) * self.res_weight
        return self.dropout2(x)

class RelativeScaledDotSAMTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int,
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None,
                 normtype="LayerNorm") -> None:
        super().__init__()
        self.layer = RelativeCustomSAMTransformerEncoderLayer(d_model, nhead, 
                                                      n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                                                      dim_feedforward, dropout,
                                                      activation,
                                                      layer_norm_eps, batch_first, norm_first,
                                                      bias, device, dtype, RelativeScaledDotSAMMultiHeadAttention, normtype)

    def forward(
            self,
            src: Tensor,
            position,
            mask=None) -> Tensor:
        return self.layer(src, position, mask)
    
class RelativeEuclideanSAMTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int,
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None,
                 normtype="LayerNorm") -> None:
        super().__init__()
        self.layer = RelativeCustomSAMTransformerEncoderLayer(d_model, nhead, 
                                                      n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                                                      dim_feedforward, dropout,
                                                      activation,
                                                      layer_norm_eps, batch_first, norm_first,
                                                      bias, device, dtype, RelativeEuclideanSAMMultiHeadAttention, normtype)

    def forward(
            self,
            src: Tensor,
            position,
            mask=None) -> Tensor:
        return self.layer(src, position, mask)
    
class RelativeCompositeSAMTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int,
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None,
                 normtype="LayerNorm") -> None:
        super().__init__()
        self.layer = RelativeCustomSAMTransformerEncoderLayer(d_model, nhead, 
                                                      n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                                                      dim_feedforward, dropout,
                                                      activation,
                                                      layer_norm_eps, batch_first, norm_first,
                                                      bias, device, dtype, RelativeCompositeSAMMultiHeadAttention, normtype)

    def forward(
            self,
            src: Tensor,
            position,
            mask=None) -> Tensor:
        return self.layer(src, position, mask)