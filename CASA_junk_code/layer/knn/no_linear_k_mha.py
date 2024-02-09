import torch
import torch.nn as nn
from knn_cuda import KNN
from layer.utils import get_neighbors
from layer.attention import Attention
from layer.utils import scale_dot_score

class NoLinearQKMHA(nn.Module):
    def __init__(self, d_model, n_head, add_zero_attn=False):
        super().__init__()
        self.n_head = n_head
        self.attention = Attention(n_head, scale_dot_score, False, False)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_q = nn.Identity()
        self.w_k = nn.Identity()
        self.w_concat = nn.Linear(d_model, d_model)

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