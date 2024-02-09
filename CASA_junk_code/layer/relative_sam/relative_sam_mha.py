import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layer.utils import GRSoftmax, euclidean_score, scale_dot_score, composite_score
from layer.samsa.sam import SAM
from layer.norm import BatchNormNear, SequenceFeatureNorm, BatchNormMean
from typing import List, Union, Any, Callable, Optional
from layer.utils import _get_activation_fn
from layer.utils.activation import LearnableReverseReLU
import math

def pairwise_distances(x, y):
    '''
    Input: x is a BxNxd matrix
           y is an optional BxMxd matirx
    Output: dist is a BxNxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    with torch.no_grad():
        x_norm = (x**2).sum(2).unsqueeze(2)
        y_t = torch.transpose(y, 1, 2)
        y_norm = (y**2).sum(2).unsqueeze(1)

        dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
        return dist

def pairwise_minus(x, y):
    '''
    Input: x is a BxNxd matrix
           y is an optional BxMxd matirx
    Output: dist is a BxNxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    return x.unsqueeze(2) - y.unsqueeze(1)

def dot_product(x, y):
    return torch.bmm(x, y.transpose(1, 2)) / math.sqrt(x.shape[-1])

class ffn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.LazyLinear(dim * 4),
                                 nn.GELU(),
                                 nn.LazyLinear(dim))

    def forward(self, x):
        return self.mlp(x)

class LearnableRelativeAttentionBound(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1, 1, 1, 1), requires_grad=True)
    
    def forward(self, x):
        return torch.minimum(x, F.softplus(self.param))

class RelativeSAMAttention(nn.Module):
    def __init__(self, nhead, score_functions, norm=False, hard=False, drop_global_points=False):
        super().__init__()
        self.drop_global_points = drop_global_points
        self.n_global_points = None
        self.score_functions = score_functions(norm, nhead)
        self.linear_b = nn.LazyLinear(nhead)
        self.linear_w = nn.LazyLinear(nhead)
        # self.relact_b = LearnableReverseReLU()
        # self.relact_w = LearnableReverseReLU()
        self.relact_b = nn.Identity()
        self.relact_w = nn.Identity()
        # self.norm = BatchNormNear(1)
        # self.norm = nn.BatchNorm1d(1, affine=False)
        self.norm = nn.Identity()
        
    def forward(self, q, k, v, c, d_q, d_k, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        if d_q.shape[2] == 4:
            distmap = pairwise_distances(d_q[:,:,:3], d_k[:,:,:3]).unsqueeze(-1)
            distmap = self.norm(distmap.reshape(-1, 1)).reshape(distmap.shape)
            distmap_w = F.softplus(self.relact_w(self.linear_w(distmap) * -1).permute(0, 3, 1, 2))
            distmap_b = self.relact_b(self.linear_b(distmap)).permute(0, 3, 1, 2)
        else:
            distmap1 = pairwise_distances(d_q[:,:,:3], d_k[:,:,:3]).unsqueeze(-1)
            distmap1 = self.norm(distmap1.reshape(-1, 1)).reshape(distmap1.shape)
            distmap2 = dot_product(d_q[:,:,3:6], d_k[:,:,3:6]).unsqueeze(-1)
            distmap = torch.cat([distmap1, distmap2], dim=-1)
            distmap_w = F.softplus(self.relact_w(self.linear_w(distmap) * -1).permute(0, 3, 1, 2))
            distmap_b = self.relact_b(self.linear_b(distmap)).permute(0, 3, 1, 2)
        scores = self.score_functions(q, k)
        scores = scores * distmap_w + distmap_b
        scores = F.relu(scores)
        detr = torch.sum(scores, dim=-1, keepdim=True) + torch.sum(F.softplus(c), dim=2, keepdim=True) + 1e-9
        scores = scores / detr
        v = scores @ v
        return v, scores

class RelativeSAMCMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, score_functions, score_norm=False, transformation_qk=True, transformation_v=True, add_zero_attn=False, hard=False, drop_global_points=False):
        super().__init__()
        self.n_head = n_head
        self.attention = RelativeSAMAttention(n_head, score_functions, score_norm, hard, drop_global_points)
        
        self.w_c = nn.Linear(d_model, n_head)

        if transformation_qk == True:
            self.w_q = nn.Linear(d_model, n_head)
            self.w_k = nn.Linear(d_model, n_head)
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
        
    def forward(self, q, k, v, d_q, d_k, mask=None):
        if self.add_zero_attn == True:
            zeros = torch.zeros(k.shape[0], 1, self.d_model, device=q.device)
            k = torch.cat([k, zeros], dim=1)
            v = torch.cat([v, zeros], dim=1)

        # 1. dot product with weight matrices
        q, k, v, c = self.w_q(q), self.w_k(k), self.w_v(v), self.w_c(v)

        # 2. split tensor by number of heads
        q, k, v, c = self.split(q), self.split(k), self.split(v), self.split(c)

        # 3. do scale euclidean attention to compute similarity
        out, attention = self.attention(q, k, v, c, d_q, d_k, mask=mask)

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

class RelativeSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                 score_functions, add_zero_attn=False, transformation_qk=True, 
                 score_norm=False, hard=False):
        super().__init__()

        self.SAM = SAM(n_sampled_points_lb, n_sampled_points_ub, d_model, drop_point)

        self.att = RelativeSAMCMultiHeadAttention(d_model, 
                                       n_head, 
                                       score_functions=score_functions,
                                       transformation_qk=transformation_qk, 
                                       add_zero_attn=add_zero_attn,
                                       hard=hard,
                                       score_norm=score_norm,
                                       drop_global_points=drop_global_points)
        
        self.d_model = d_model
        
    def forward(self, q, k, v, d_q, mask=None):
        vdq = torch.cat([v, d_q], dim=-1)
        kvdq, regularization = self.SAM(k, vdq, mask=mask)
        k, v, d_k = kvdq[:,:,:self.d_model], kvdq[:,:,self.d_model:2*self.d_model], kvdq[:,:,2*self.d_model:]
        out, attention = self.att(q, k, v, d_q, d_k, None)
        return out, attention, regularization
    
class RelativeEuclideanSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                 add_zero_attn=False, hard=False):
        super().__init__()
        self.att = RelativeSAMMultiHeadAttention(d_model, 
                                         n_head, 
                                         n_sampled_points_lb,
                                         n_sampled_points_ub,
                                         drop_point,
                                         drop_global_points=drop_global_points,
                                         score_functions=euclidean_score, 
                                         score_norm=True, 
                                         transformation_qk=True, 
                                         add_zero_attn=add_zero_attn,
                                         hard=hard)
        
    def forward(self, q, k, v, d_q, mask=None):
        out, attention, regularization = self.att(q, k, v, d_q, mask)
        return out, attention, regularization
    
class RelativeScaledDotSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                 add_zero_attn=False, hard=False):
        super().__init__()
        self.att = RelativeSAMMultiHeadAttention(d_model, 
                                         n_head, 
                                         n_sampled_points_lb,
                                         n_sampled_points_ub,
                                         drop_point,
                                         drop_global_points=drop_global_points,
                                         score_functions=scale_dot_score, 
                                         score_norm=False, 
                                         transformation_qk=True, 
                                         add_zero_attn=add_zero_attn,
                                         hard=hard)
        
    def forward(self, q, k, v, d_q, mask=None):
        out, attention, regularization = self.att(q, k, v, d_q, mask)
        return out, attention, regularization
    
class RelativeCompositeSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                 add_zero_attn=False, hard=False):
        super().__init__()
        self.att = RelativeSAMMultiHeadAttention(d_model, 
                                         n_head, 
                                         n_sampled_points_lb,
                                         n_sampled_points_ub,
                                         drop_point,
                                         drop_global_points=drop_global_points,
                                         score_functions=composite_score, 
                                         score_norm=False, 
                                         transformation_qk=True, 
                                         add_zero_attn=add_zero_attn,
                                         hard=hard)
        
    def forward(self, q, k, v, d_q, mask=None):
        out, attention, regularization = self.att(q, k, v, d_q, mask)
        return out, attention, regularization
    
class RelativeNeuralSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                 add_zero_attn=False, hard=False):
        super().__init__()
        self.att = RelativeSAMMultiHeadAttention(d_model, 
                                         n_head, 
                                         n_sampled_points_lb,
                                         n_sampled_points_ub,
                                         drop_point,
                                         drop_global_points=drop_global_points,
                                         score_functions='neural', 
                                         score_norm=False, 
                                         transformation_qk=True, 
                                         add_zero_attn=add_zero_attn,
                                         hard=hard)
        
    def forward(self, q, k, v, d_q, mask=None):
        out, attention, regularization = self.att(q, k, v, d_q, mask)
        return out, attention, regularization