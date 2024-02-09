import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layer.utils import GRSoftmax, euclidean_score, scale_dot_score, composite_score
from layer.samsa.sam import SAM
from typing import List, Union, Any, Callable, Optional
from layer.utils import _get_activation_fn
import math

# class ScoreNorm(nn.Module):
#     def __init__(self, nhead, eps=1e-5, momentum=0.1):
#         super().__init__()
#         self.nhead = nhead
#         self.eps = eps
#         self.momentum = momentum

#         # Initialize learnable parameters
#         self.gamma = nn.Parameter(torch.ones(1, nhead, 1, 1))

#     def forward(self, x):
#         if self.training:
#             # Calculate batch statistics during training
#             batch_mean = x.mean(dim=(0, 2), keepdim=True)
#             batch_var = x.var(dim=(0, 2), unbiased=False, keepdim=True)
            
#             # Update running statistics with momentum
#             self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
#             self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

#             # Normalize the input
#             x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
#         else:
#             # Use running statistics during evaluation
#             x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

#         # Scale and shift
#         y = self.gamma * x_normalized + self.beta

#         return y

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

class SAMAttention(nn.Module):
    def __init__(self, nhead, score_functions, norm=False, hard=False):
        super().__init__()
        if hard == False:
            self.softmax = F.softmax
        else:
            self.softmax = GRSoftmax(temp=0.1, k=3)
            
        self.score_functions = score_functions(norm, nhead)
        self.constant = nn.Parameter(torch.ones(1, nhead, 1, 1), requires_grad=True)
        
    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        if mask is None:
            scores = F.relu(self.score_functions(q, k))
            detr = torch.sum(scores, dim=-1, keepdim=True) + v.shape[2] * F.softplus(self.constant)
            detr = detr + torch.sign(detr) * 1e-9
            scores = scores / detr
        else:
            mask = mask.unsqueeze(1) # Input mask = (batch, n_seq, 1)
            mask = mask.unsqueeze(1)
            scores = F.relu(self.score_functions(q, k) + mask * (-1e9))
            detr = torch.sum(scores, dim=-1, keepdim=True) + v.shape[2] * self.constant
            detr = detr + torch.sign(detr) * 1e-9
            scores = scores / detr
        v = scores @ v
        return v, scores

class SAMAttention(nn.Module):
    def __init__(self, nhead, score_functions, norm=False, hard=False, leaky_softmax=False):
        super().__init__()
        if hard == False:
            self.softmax = F.softmax
        else:
            self.softmax = GRSoftmax(temp=0.1, k=3)
            
        self.score_functions = score_functions(norm, nhead)
        self.constant = nn.Parameter(torch.randn(1, nhead, 1, 1), requires_grad=True)

        if leaky_softmax == False:
            self.forward = self.forward_noleaky
        else:
            self.forward = self.forward_leaky

    def forward_noleaky(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        if mask is None:
            scores = self.score_functions(q, k)
            scores = F.softmax(scores, dim=-1)
        else:
            mask = mask.unsqueeze(1) # Input mask = (batch, n_seq, 1)
            mask = mask.unsqueeze(1)
            scores = self.score_functions(q, k) + mask * (-1e9)
            scores = softmax(scores, dim=-1)
        v = scores @ v
        return v, scores

    def forward_leaky(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        if mask is None:
            scores = self.score_functions(q, k)
            scores = softmax(scores, self.constant, dim=-1)
        else:
            mask = mask.unsqueeze(1) # Input mask = (batch, n_seq, 1)
            mask = mask.unsqueeze(1)
            scores = self.score_functions(q, k) + mask * (-1e9)
            scores = softmax(scores, self.constant, dim=-1)
        v = scores @ v
        return v, scores

class SAMCMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, score_functions, score_norm=False, transformation_qk=True, transformation_v=True, add_zero_attn=False, hard=False, leaky_softmax=False):
        super().__init__()
        self.n_head = n_head
        self.attention = SAMAttention(n_head, score_functions, score_norm, hard, leaky_softmax)
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

class SAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, leaky_softmax,
                 score_functions, add_zero_attn=False, transformation_qk=True, 
                 score_norm=False, hard=False):
        super(SAMMultiHeadAttention, self).__init__()

        self.SAM = SAM(n_sampled_points_lb, n_sampled_points_ub, d_model, drop_point)

        self.att = SAMCMultiHeadAttention(d_model, 
                                       n_head, 
                                       score_functions=score_functions,
                                       transformation_qk=transformation_qk, 
                                       add_zero_attn=add_zero_attn,
                                       hard=hard,
                                       score_norm=score_norm,
                                       leaky_softmax=leaky_softmax)
        
        self.d_model = d_model
        
    def forward(self, q, k, v, mask=None):
        kv = self.SAM(k, v, mask=mask)
        k, v = kv[:,:,:self.d_model], kv[:,:,self.d_model:]
        out, attention = self.att(q, k, v, None)
        return out, attention
    
class EuclideanSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, leaky_softmax,
                 add_zero_attn=False, hard=False):
        super(EuclideanSAMMultiHeadAttention, self).__init__()
        self.att = SAMMultiHeadAttention(d_model, 
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
        
    def forward(self, q, k, v, mask=None):
        out, attention = self.att(q, k, v, mask)
        return out, attention
    
class ScaledDotSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, leaky_softmax,
                 add_zero_attn=False, hard=False):
        super(ScaledDotSAMMultiHeadAttention, self).__init__()
        self.att = SAMMultiHeadAttention(d_model, 
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
        
    def forward(self, q, k, v, mask=None):
        out, attention = self.att(q, k, v, mask)
        return out, attention
    
class CompositeSAMMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, 
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, leaky_softmax,
                 add_zero_attn=False, hard=False):
        super(CompositeSAMMultiHeadAttention, self).__init__()
        self.att = SAMMultiHeadAttention(d_model, 
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
        
    def forward(self, q, k, v, mask=None):
        out, attention = self.att(q, k, v, mask)
        return out, attention