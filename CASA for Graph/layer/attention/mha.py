import torch
import torch.nn as nn
import torch.nn.functional as F
from functional import PSAM

class Attention(nn.Module):
    def __init__(self, nhead, score_functions, relative_function, probability_function):
        super().__init__()
        self.score_functions = score_functions
        self.relative_function = relative_function
        self.probability_function = probability_function
        self.is_leaky = probability_function.is_leaky
        self.linear_w = nn.LazyLinear(nhead)
        self.linear_b = nn.LazyLinear(nhead)
        
    def forward(self, q, k, v, c, d_q, d_k_top, d_k_bot, d_k_score):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        scores = self.score_functions(q, k)

        if self.relative_function is not None:
            distmap_top = self.relative_function(d_q, d_k_top) # b,l1,l2,d
            distmap_bot = self.relative_function(d_q, d_k_bot) # b,l1,l2,d
            distmap = torch.stack([distmap_top, distmap_bot], dim=3) # b,l1,l2,1,d
            distmap = distmap * d_k_score.unsqueeze(1) # b,1,l2,2,1
            distmap = torch.sum(distmap, dim=3)
            distmap_w = F.softplus(self.linear_w(distmap))
            distmap_b = self.linear_b(distmap)
            distmap_w, distmap_b = distmap_w.permute(0, 3, 1, 2), distmap_b.permute(0, 3, 1, 2)
            scores = scores * distmap_w + distmap_b

        if self.is_leaky is True:
            scores = self.probability_function(scores, c)
        else:
            scores = self.probability_function(scores)
        v = scores @ v
        return v, scores

class CMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, score_functions, relative_function, probability_function):
        super().__init__()
        self.nhead = nhead
        self.attention = Attention(nhead, score_functions, relative_function, probability_function)
        
        self.w_c = nn.Linear(d_model, nhead)
        self.w_q = nn.Linear(d_model, nhead)
        self.w_k = nn.Linear(d_model, nhead)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_concat = nn.Linear(d_model, d_model)
        self.d_model = d_model
        
    def forward(self, q, k, v, d_q, d_k_top, d_k_bot, d_k_score):
        q, k, v, c = self.w_q(q), self.w_k(k), self.w_v(v), self.w_c(v)

        q, k, v, c = self.split(q), self.split(k), self.split(v), self.split(c)

        out, attention = self.attention(q, k, v, c, d_q, d_k_top, d_k_bot, d_k_score)

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

        d_tensor = d_model // self.nhead
        tensor = tensor.view(batch_size, length, self.nhead, d_tensor).transpose(1, 2)
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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, 
                 n_sampled_points, drop_point,
                 score_functions, relative_function, probability_function):
        super().__init__()

        self.SAM = PSAM(n_sampled_points, d_model, drop_point)

        self.att = CMultiHeadAttention(d_model, 
                                       nhead, 
                                       score_functions=score_functions,
                                       relative_function=relative_function,
                                       probability_function=probability_function)
        
        self.d_model = d_model
        
    def forward(self, q, k, v, d_q, relative_map, mask=None):
        vdq = torch.cat([v, d_q], dim=-1)
        kvdq, kvdq_top, kvdq_bottom, x_score = self.SAM(k, vdq, mask=mask)
        k, v = kvdq[:,:,:self.d_model], kvdq[:,:,self.d_model:2*self.d_model]
        d_k_top, d_k_bottom = kvdq_top[:,:,2*self.d_model:], kvdq_bottom[:,:,2*self.d_model:]
        out, attention = self.att(q, k, v, d_q, d_k_top, d_k_bottom, x_score)
        return out, attention