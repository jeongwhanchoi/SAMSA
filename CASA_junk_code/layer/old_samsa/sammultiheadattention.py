import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layer.utils import GRSoftmax, euclidean_score, scale_dot_score, composite_score
from layer.samsa.sam import SAM
from layer.attention import CMultiHeadAttention
from typing import List, Union, Any, Callable, Optional
from layer.utils import _get_activation_fn

class SAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, reduced_transform,
                 score_functions, add_zero_attn=False, transformation_qk=True, 
                 score_norm=False, hard=False):
        super(SAMMultiHeadAttention, self).__init__()

        self.SAM = SAM(n_sampled_points_lb, n_sampled_points_ub, 1, d_model, drop_point)

        self.att = CMultiHeadAttention(d_model, 
                                       n_head, 
                                       score_functions=score_functions,
                                       transformation_qk=transformation_qk, 
                                       add_zero_attn=add_zero_attn,
                                       hard=hard,
                                       score_norm=score_norm)

        if reduced_transform == False:
            self.transf_v = nn.Identity()
        else:
            self.transf_v = nn.TransformerEncoderLayer(d_model, n_head, d_model * 4, activation='gelu')
        
        self.d_model = d_model
        
    def forward(self, q, k, v, mask=None):
        k, p = self.SAM(k, mask=mask)
        v = self.transf_v(torch.bmm(p, v))
        out, attention = self.att(q, k, v, None)
        return out, attention
    
class EuclideanSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, reduced_transform,
                 add_zero_attn=False, hard=False):
        super(EuclideanSAMMultiHeadAttention, self).__init__()
        self.att = SAMMultiHeadAttention(d_model, 
                                         n_head, 
                                         n_sampled_points_lb,
                                         n_sampled_points_ub,
                                         drop_point,
                                         reduced_transform,
                                         score_functions=euclidean_score, 
                                         score_norm=True, 
                                         transformation_qk=True, 
                                         add_zero_attn=add_zero_attn,
                                         hard=hard)
        
    def forward(self, q, k, v, mask=None):
        out, attention = self.att(q, k, v, mask)
        return out, attention
    
class ScaledDotSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, reduced_transform,
                 add_zero_attn=False, hard=False):
        super(ScaledDotSAMMultiHeadAttention, self).__init__()
        self.att = SAMMultiHeadAttention(d_model, 
                                         n_head, 
                                         n_sampled_points_lb,
                                         n_sampled_points_ub,
                                         drop_point,
                                         reduced_transform,
                                         score_functions=scale_dot_score, 
                                         score_norm=False, 
                                         transformation_qk=True, 
                                         add_zero_attn=add_zero_attn,
                                         hard=hard)
        
    def forward(self, q, k, v, mask=None):
        out, attention = self.att(q, k, v, mask)
        return out, attention
    
class CompositeSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, reduced_transform,
                 add_zero_attn=False, hard=False):
        super(CompositeSAMMultiHeadAttention, self).__init__()
        self.att = SAMMultiHeadAttention(d_model, 
                                         n_head, 
                                         n_sampled_points_lb,
                                         n_sampled_points_ub,
                                         drop_point,
                                         reduced_transform,
                                         score_functions=composite_score, 
                                         score_norm=True, 
                                         transformation_qk=True, 
                                         add_zero_attn=add_zero_attn,
                                         hard=hard)
        
    def forward(self, q, k, v, mask=None):
        out, attention = self.att(q, k, v, mask)
        return out, attention