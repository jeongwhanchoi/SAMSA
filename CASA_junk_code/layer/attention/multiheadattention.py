import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layer.utils import GRSoftmax, euclidean_score, scale_dot_score, composite_score
from typing import List, Union, Any, Callable, Optional
from layer.utils import _get_activation_fn

class Attention(nn.Module):
    def __init__(self, nhead, score_functions, norm=False, hard=False):
        super(Attention, self).__init__()
        if hard == False:
            self.softmax = F.softmax
        else:
            self.softmax = GRSoftmax(temp=0.1, k=3)
            
        self.score_functions = score_functions(norm, nhead)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        if mask is None:
            scores = self.softmax(self.score_functions(q, k), dim=3)
        else:
            mask = mask.unsqueeze(1) # Input mask = (batch, n_seq, 1)
            mask = mask.unsqueeze(1)
            scores = self.softmax(self.score_functions(q, k) + mask * (-1e9), dim=3)
        v = scores @ v
        return v, scores

class CMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, score_functions, score_norm=False, transformation_qk=True, transformation_v=True, add_zero_attn=False, hard=False):
        super(CMultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = Attention(n_head, score_functions, score_norm, hard)
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

class EuclideanMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, add_zero_attn=False, hard=False):
        super(EuclideanMultiHeadAttention, self).__init__()
        self.att = CMultiHeadAttention(d_model, 
                                       n_head, 
                                       score_functions=euclidean_score, 
                                       score_norm=True, 
                                       transformation_qk=True, 
                                       add_zero_attn=add_zero_attn,
                                       hard=hard)
        
    def forward(self, q, k, v, mask=None):
        out, attention = self.att(q, k, v, mask)
        return out, attention

class ScaledDotMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, add_zero_attn=False, hard=False):
        super(ScaledDotMultiHeadAttention, self).__init__()
        self.att = CMultiHeadAttention(d_model, 
                                       n_head, 
                                       score_functions=scale_dot_score, 
                                       score_norm=False, 
                                       transformation_qk=True, 
                                       add_zero_attn=add_zero_attn,
                                       hard=hard)
        
    def forward(self, q, k, v, mask=None):
        out, attention = self.att(q, k, v, mask)
        return out, attention
    
class CompositeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, add_zero_attn=False, hard=False):
        super(CompositeMultiHeadAttention, self).__init__()
        self.att = CMultiHeadAttention(d_model, 
                                       n_head, 
                                       score_functions=composite_score, 
                                       score_norm=False, 
                                       transformation_qk=True, 
                                       add_zero_attn=add_zero_attn,
                                       hard=hard)
        
    def forward(self, q, k, v, mask=None):
        out, attention = self.att(q, k, v, mask)
        return out, attention