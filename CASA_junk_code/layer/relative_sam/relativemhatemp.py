import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layer.utils import GRSoftmax, euclidean_score, scale_dot_score, composite_score
from layer.samsa.sam import SAM
from typing import List, Union, Any, Callable, Optional
from layer.utils import _get_activation_fn
import math

class BatchNormNear(nn.Module):
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        super(BatchNormNear, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        # Moving averages for mean and variance
        self.register_buffer('running_mean', torch.zeros(num_features))

        self.calculated = False

    def forward(self, x):
        if self.training:
            mean = torch.min(x, dim=0, keepdim=True)[0]
            mean = torch.mean(mean, dim=[i for i in range(len(mean.shape) - 1)], keepdim=True)

            if self.calculated == True:
                # Update running mean and variance with momentum
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            else:
                self.running_mean = mean
                self.calculated = True

        x_normalized = (x - self.running_mean) / self.running_mean
        return x_normalized

def softmax(x, leaky_factor, dim=1):
    """
    Compute the softmax function for each row along the specified dimension of the input x.

    Arguments:
    x: A PyTorch tensor.
    dim: The dimension along which the softmax will be computed. Default is 1.

    Returns:
    A PyTorch tensor containing the softmax probabilities along the specified dimension.
    """
    max_factor = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - max_factor)
    leaky_factor = torch.exp(leaky_factor - max_factor) * x.shape[dim]
    softmax_probs = exp_x / (torch.sum(exp_x, dim=dim, keepdim=True) + leaky_factor)
    return softmax_probs

def pairwise_distances(x, y):
    '''
    Input: x is a BxNxd matrix
           y is an optional BxMxd matirx
    Output: dist is a BxNxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(2).unsqueeze(2)
    y_t = torch.transpose(y, 1, 2)
    y_norm = (y**2).sum(2).unsqueeze(1)

    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    return dist

def dot_product(x, y):
    return torch.bmm(x, y.transpose(1, 2)) / math.sqrt(x.shape[-1])

class res_norm_mlp(nn.Module):
    def __init__(self, n_d):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(n_d * 4),
                                 nn.GELU(),
                                 nn.LazyLinear(n_d))
        self.linear = nn.LazyLinear(n_d)
        self.norm = nn.LayerNorm(n_d)
    
    def forward(self, x):
        return self.norm(self.net(x) + self.linear(x))
        

class RelativeSAMAttention(nn.Module):
    def __init__(self, nhead, d_model, score_functions, norm=False, hard=False, leaky_softmax=False):
        super().__init__()
        if hard == False:
            self.softmax = F.softmax
        else:
            self.softmax = GRSoftmax(temp=0.1, k=3)
        
        if isinstance(score_functions, str):
            if score_functions == 'neural':
                self.score_functions = res_norm_mlp(nhead)
                self.forward = self.forward_neural
        else:
            self.score_functions = score_functions(norm, nhead)
            self.linearq = nn.LazyLinear(nhead)
            self.lineark = nn.LazyLinear(nhead)
            self.linearv = nn.LazyLinear(d_model)
            if leaky_softmax == False:
                self.forward = self.forward_noleaky
            else:
                self.forward = self.forward_leaky
        
        self.norm = BatchNormNear(1)
        self.nhead = nhead
        self.d_m = d_model // nhead
        
    def forward_neural(self, q, k, v, c, d_q, d_k, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        if d_q.shape[2] == 6:
            d_map = pairwise_distances(d_q[:,:,:3], d_k[:,:,:3]).unsqueeze(1).unsqueeze(-1) + torch.zeros(q.shape[0], 1, q.shape[2], k.shape[2], 1, device=q.device)
            d_map = self.norm(d_map)
            n_map = dot_product(d_q[:,:,:3], d_k[:,:,:3]).unsqueeze(1).unsqueeze(-1) + torch.zeros(q.shape[0], 1, q.shape[2], k.shape[2], 1, device=q.device)
            q_exp = q.unsqueeze(3)
            k_exp = k.unsqueeze(2)
            q_exp, k_exp = q_exp + torch.zeros_like(k_exp), k_exp + torch.zeros_like(q_exp) # b, h, l1, l2, 1
            q_exp, k_exp = q_exp.transpose(1, 4), k_exp.transpose(1, 4)
            a_map = torch.cat([q_exp, k_exp, d_map, n_map], dim=4)
            a_map = self.score_functions(a_map).transpose(1, 4).squeeze(-1)
        else:
            d_map = pairwise_distances(d_q, d_k).unsqueeze(1).unsqueeze(-1) + torch.zeros(q.shape[0], 1, q.shape[2], k.shape[2], 1, device=q.device)
            d_map = self.norm(d_map)
            q_exp = q.unsqueeze(3)
            k_exp = k.unsqueeze(2)
            q_exp, k_exp = q_exp + torch.zeros_like(k_exp), k_exp + torch.zeros_like(q_exp) # b, h, l1, l2, 1
            q_exp, k_exp = q_exp.transpose(1, 4), k_exp.transpose(1, 4)
            a_map = torch.cat([q_exp, k_exp, d_map], dim=4)
            a_map = self.score_functions(a_map).transpose(1, 4).squeeze(-1)
        scores = F.relu(a_map)
        detr = torch.sum(scores, dim=-1, keepdim=True) + torch.sum(F.softplus(c), dim=2, keepdim=True) + 1e-9
        scores = scores / detr
        v = scores @ v
        return v, scores

    def forward_noleaky(self, q, k, v, c, d_q, d_k, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        b, l1, l2, d = q.shape[0], q.shape[2], k.shape[2], v.shape[3]
        if d_q.shape[2] == 3:
            distmap = self.norm(pairwise_distances(d_q, d_k).unsqueeze(-1))
            distmapq, distmapk = self.linearq(distmap).permute(0, 3, 1, 2), self.lineark(distmap).permute(0, 3, 1, 2)
            distmapv = self.linearv(distmap).reshape(b, l1, l2, self.nhead, d).permute(0, 3, 1, 2, 4)
            distmap = torch.stack([distmapq, distmapk], dim=-1)
        else:
            distmap1 = self.norm(pairwise_distances(d_q[:,:,:3], d_k[:,:,:3]).unsqueeze(-1))
            distmap2 = dot_product(d_q[:,:,3:], d_k[:,:,3:]).unsqueeze(-1)
            distmap = torch.cat([distmap1, distmap2], dim=-1)
            distmapq, distmapk = self.linearq(distmap).permute(0, 3, 1, 2), self.lineark(distmap).permute(0, 3, 1, 2)
            distmapv = self.linearv(distmap).reshape(b, l1, l2, self.nhead, d).permute(0, 3, 1, 2, 4)
            distmap = torch.stack([distmapq, distmapk], dim=-1)
        scores = self.score_functions(q, k, distmap)
        detr = torch.sum(F.softplus(c), dim=2, keepdim=True) + 1e-9
        scores = scores / detr
        v = v.unsqueeze(2) + distmapv
        v = torch.sum(scores.unsqueeze(-1) * v, dim=3)
        return v, scores
        
    def forward_leaky(self, q, k, v, c, d_q, d_k, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        b, l1, l2, d = q.shape[0], q.shape[2], k.shape[2], v.shape[3]
        if d_q.shape[2] == 3:
            distmap = self.norm(pairwise_distances(d_q, d_k).unsqueeze(-1))
            distmapq, distmapk = self.linearq(distmap).permute(0, 3, 1, 2), self.lineark(distmap).permute(0, 3, 1, 2)
            distmapv = self.linearv(distmap).reshape(b, l1, l2, self.nhead, d).permute(0, 3, 1, 2, 4)
            distmap = torch.stack([distmapq, distmapk], dim=-1)
        else:
            distmap1 = self.norm(pairwise_distances(d_q[:,:,:3], d_k[:,:,:3]).unsqueeze(-1))
            distmap2 = dot_product(d_q[:,:,3:], d_k[:,:,3:]).unsqueeze(-1)
            distmap = torch.cat([distmap1, distmap2], dim=-1)
            distmapq, distmapk = self.linearq(distmap).permute(0, 3, 1, 2), self.lineark(distmap).permute(0, 3, 1, 2)
            distmapv = self.linearv(distmap).reshape(b, l1, l2, self.nhead, d).permute(0, 3, 1, 2, 4)
            distmap = torch.stack([distmapq, distmapk], dim=-1)
        scores = self.score_functions(q, k, distmap)
        scores = F.relu(scores)
        detr = torch.sum(scores, dim=-1, keepdim=True) + torch.sum(F.softplus(c), dim=2, keepdim=True) + 1e-9
        scores = scores / detr
        v = v.unsqueeze(2) + distmapv
        v = torch.sum(scores.unsqueeze(-1) * v, dim=3)
        return v, scores

class RelativeSAMCMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, score_functions, score_norm=False, transformation_qk=True, transformation_v=True, add_zero_attn=False, hard=False, leaky_softmax=False):
        super().__init__()
        self.n_head = n_head
        self.attention = RelativeSAMAttention(n_head, d_model, score_functions, score_norm, hard, leaky_softmax)
        
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
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, leaky_softmax,
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
                                       leaky_softmax=leaky_softmax)
        
        self.d_model = d_model
        
    def forward(self, q, k, v, d_q, mask=None):
        vdq = torch.cat([v, d_q], dim=-1)
        kvdq = self.SAM(k, vdq, mask=mask)
        k, v, d_k = kvdq[:,:,:self.d_model], kvdq[:,:,self.d_model:2*self.d_model], kvdq[:,:,2*self.d_model:]
        out, attention = self.att(q, k, v, d_q, d_k, None)
        return out, attention
    
class RelativeEuclideanSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, leaky_softmax,
                 add_zero_attn=False, hard=False):
        super().__init__()
        self.att = RelativeSAMMultiHeadAttention(d_model, 
                                         n_head, 
                                         n_sampled_points_lb,
                                         n_sampled_points_ub,
                                         drop_point,
                                         leaky_softmax=leaky_softmax,
                                         score_functions=euclidean_score, 
                                         score_norm=True, 
                                         transformation_qk=True, 
                                         add_zero_attn=add_zero_attn,
                                         hard=hard)
        
    def forward(self, q, k, v, d_q, mask=None):
        out, attention = self.att(q, k, v, d_q, mask)
        return out, attention
    
class RelativeScaledDotSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, leaky_softmax,
                 add_zero_attn=False, hard=False):
        super().__init__()
        self.att = RelativeSAMMultiHeadAttention(d_model, 
                                         n_head, 
                                         n_sampled_points_lb,
                                         n_sampled_points_ub,
                                         drop_point,
                                         leaky_softmax=leaky_softmax,
                                         score_functions=scale_dot_score, 
                                         score_norm=False, 
                                         transformation_qk=True, 
                                         add_zero_attn=add_zero_attn,
                                         hard=hard)
        
    def forward(self, q, k, v, d_q, mask=None):
        out, attention = self.att(q, k, v, d_q, mask)
        return out, attention
    
class RelativeCompositeSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, leaky_softmax,
                 add_zero_attn=False, hard=False):
        super().__init__()
        self.att = RelativeSAMMultiHeadAttention(d_model, 
                                         n_head, 
                                         n_sampled_points_lb,
                                         n_sampled_points_ub,
                                         drop_point,
                                         leaky_softmax=leaky_softmax,
                                         score_functions=composite_score, 
                                         score_norm=False, 
                                         transformation_qk=True, 
                                         add_zero_attn=add_zero_attn,
                                         hard=hard)
        
    def forward(self, q, k, v, d_q, mask=None):
        out, attention = self.att(q, k, v, d_q, mask)
        return out, attention
    
class RelativeNeuralSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, leaky_softmax,
                 add_zero_attn=False, hard=False):
        super().__init__()
        self.att = RelativeSAMMultiHeadAttention(d_model, 
                                         n_head, 
                                         n_sampled_points_lb,
                                         n_sampled_points_ub,
                                         drop_point,
                                         leaky_softmax=leaky_softmax,
                                         score_functions='neural', 
                                         score_norm=False, 
                                         transformation_qk=True, 
                                         add_zero_attn=add_zero_attn,
                                         hard=hard)
        
    def forward(self, q, k, v, d_q, mask=None):
        out, attention = self.att(q, k, v, d_q, mask)
        return out, attention