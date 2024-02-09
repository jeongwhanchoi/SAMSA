import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from layer.utils import GRSoftmax, euclidean_score, scale_dot_score, composite_score
import math

class SoftplusPWP(nn.Module):
    def __init__(self, n_hidden, attention_dimension):
        super().__init__()
        self.score_functions = composite_score(False, attention_dimension)
        self.linear_q = nn.Linear(n_hidden, attention_dimension)
        self.linear_k = nn.Linear(n_hidden, attention_dimension)
        self.linear_v = nn.Linear(n_hidden, n_hidden)
        self.attention_dimension = attention_dimension
        self.n_hidden = n_hidden
        self.norm = nn.LayerNorm(n_hidden)
        self.norm2 = nn.LayerNorm(n_hidden)
        self.zero_factor = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.angle = nn.Parameter(torch.randn(1, nhead, 1, 1, 1), requires_grad=True)

    def forward(self, x, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        x = self.norm(x)
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        b, l, d = v.shape
        q, k = q.unsqueeze(1), k.unsqueeze(1)
        q, k, v = q.transpose(1, 3), k.transpose(1, 3), self.split(v)

        q_, k_ = torch.unsqueeze(q, 3), torch.unsqueeze(k, 2)
        qk = (q + torch.zeros_like(q)) + (k + torch.zeros_like(k)) * 1j
        qk = (torch.cos(angle) + torch.sin(angle) * 1j) * qk
        qk = torch.stack([qk.real(), qk.imag()], dim=-1)
        qk = F.softmax(qk, dim=-1)

        # print(q.shape, k.shape)
        if mask is None:
            scores = self.score_functions(q, k)
        else:
            mask = mask.unsqueeze(1) # Input mask = (batch, n_seq, 1)
            mask = mask.unsqueeze(1)
            scores = self.score_functions(q, k) * (1 - mask)
        scores = scores / math.sqrt(l) * math.sqrt(2)
        v = x + self.concat(F.gelu(scores @ v)) * self.zero_factor
        return v

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.attention_dimension
        tensor = tensor.view(batch_size, length, self.attention_dimension, d_tensor).transpose(1, 2)
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