import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layer.utils import GRSoftmax, euclidean_score, scale_dot_score, composite_score
from typing import List, Union, Any, Callable, Optional
from layer.utils import _get_activation_fn

def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(2).unsqueeze(2)
    y_t = torch.transpose(y, 1, 2)
    y_norm = (y**2).sum(2).unsqueeze(1)

    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    return dist 

class PointSAMAttention(nn.Module):
    def __init__(self, nhead, score_functions, norm=False, hard=False):
        super().__init__()
        if hard == False:
            self.softmax = F.softmax
        else:
            self.softmax = GRSoftmax(temp=0.1, k=3)
            
        self.score_functions = score_functions(norm, nhead)
        self.relative_scorer = nn.Sequential

    def forward(self, q, k, v, p_q, p_k, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        if mask is None:
            scores = F.softplus(self.score_functions(q, k)) / v.shape[2]
        else:
            mask = mask.unsqueeze(1) # Input mask = (batch, n_seq, 1)
            mask = mask.unsqueeze(1)
            scores = F.softplus(self.score_functions(q, k) + mask * (-1e9))  / v.shape[2]
        v = scores @ v
        return v, scores

class PointSAMCMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, score_functions, score_norm=False, transformation_qk=True, transformation_v=True, add_zero_attn=False, hard=False):
        super().__init__()
        self.n_head = n_head
        self.attention = PointSAMAttention(n_head, score_functions, score_norm, hard)
        if transformation_qk == True:
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
        else:
            self.w_q = nn.Identity()
            self.w_k = nn.Identity()
            self.w_v = nn.Identity()

        if transformation_v == True:
            self.w_concat = nn.Linear(d_model, d_model)
        else:
            self.w_concat = nn.Identity()

        self.add_zero_attn = add_zero_attn
        self.d_model = d_model
        
    def forward(self, q, k, v, mask=None):
        if self.add_zero_attn == True:
            zeros = torch.zeros(k.shape[0], 1, self.d_model, device=q.device)
            k = torch.cat([k, zeros], dim=1)
            v = torch.cat([v, zeros], dim=1)

        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale euclidean attention to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        return out, attention

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class PointSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, reduced_transform,
                 score_functions, add_zero_attn=False, transformation_qk=True, 
                 score_norm=False, hard=False):
        super(PointSAMMultiHeadAttention, self).__init__()

        self.SAM = SAM(n_sampled_points_lb, n_sampled_points_ub, d_model, drop_point)

        self.att = SAMCMultiHeadAttention(d_model, 
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
        kv = self.SAM(k, v, mask=mask)
        k, v = kv[:,:,:self.d_model], kv[:,:,self.d_model:]
        out, attention = self.att(q, k, v, None)
        return out, attention
    
class EuclideanPointSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, reduced_transform,
                 add_zero_attn=False, hard=False):
        super(EuclideanPointSAMMultiHeadAttention, self).__init__()
        self.att = PointSAMMultiHeadAttention(d_model, 
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
    
class ScaledDotPointSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, reduced_transform,
                 add_zero_attn=False, hard=False):
        super(ScaledDotPointSAMMultiHeadAttention, self).__init__()
        self.att = PointSAMMultiHeadAttention(d_model, 
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
    
class CompositePointSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, reduced_transform,
                 add_zero_attn=False, hard=False):
        super(CompositePointSAMMultiHeadAttention, self).__init__()
        self.att = PointSAMMultiHeadAttention(d_model, 
                                         n_head, 
                                         n_sampled_points_lb,
                                         n_sampled_points_ub,
                                         drop_point,
                                         reduced_transform,
                                         score_functions=composite_score, 
                                         score_norm=False, 
                                         transformation_qk=True, 
                                         add_zero_attn=add_zero_attn,
                                         hard=hard)
        
    def forward(self, q, k, v, mask=None):
        out, attention = self.att(q, k, v, mask)
        return out, attention