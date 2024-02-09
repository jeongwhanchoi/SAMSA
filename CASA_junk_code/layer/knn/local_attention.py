import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN
from layer.utils import get_neighbors
from layer.knn.no_linear_k_mha import NoLinearQKMHA

class LocalAttention(nn.Module):
    def __init__(self,
                 n_k,
                 d_model,
                 d_model_qk,
                 nhead,
                 dim_feedforward,
                 dropout):
        super().__init__()
        self.linear_q = nn.Linear(d_model, d_model_qk)
        self.linear_k = nn.Linear(d_model, d_model_qk)
        self.sa = NoLinearQKMHA(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(p=dropout)
        self.ffn = nn.Sequential(nn.LayerNorm(d_model),
                                 nn.Linear(d_model, dim_feedforward),
                                 nn.GELU(),
                                 nn.Linear(dim_feedforward, d_model)
                                 )
        self.drop2 = nn.Dropout(p=dropout)
        self.residual_weight = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.knn = KNN(k=n_k, transpose_mode=True)
        self.n_k = n_k
        self.d_model_qk = d_model_qk
        self.d_model = d_model

    def forward(self, x, mask=None):
        normalized_xq, normalized_xk = F.normalize(self.linear_q(x), dim=-1), F.normalize(self.linear_k(x), dim=-1)
        if self.training == True:
            normalized_xq, normalized_xk = normalized_xq + torch.randn_like(normalized_xq, device=x.device) * 0.01, normalized_xk + torch.randn_like(normalized_xk, device=x.device) * 0.01
        if mask != None:
            mask = mask.unsqueeze(dim=2)
            normalized_xq = normalized_xq + mask * (-1e9)
            normalized_xk = normalized_xk + mask * (1e9)

        if self.n_k > x.shape[1]:
            x_v = x
            x_k = normalized_xk
            x_q = normalized_xq
        else:
            _, indx = self.knn(normalized_xk, normalized_xq)
            x_v = get_neighbors(x, indx)
            x_k = get_neighbors(normalized_xk, indx)
            x_q = normalized_xq

        b, n_s, n_k, _ = x_k.shape
        x_k = x_k.reshape(b * n_s, n_k, self.d_model_qk)
        x_v = x_v.reshape(b * n_s, n_k, self.d_model)
        x_q = x_q.reshape(b * n_s, 1, self.d_model_qk)
        x = x.reshape(b * n_s, 1, self.d_model)
        xnorm = self.norm1(x_v)
        x = x + self.drop1(self.sa(x_q, x_k, xnorm, mask)[0] * self.residual_weight)
        x = x + self.drop2(self.ffn(x) * self.residual_weight)
        x = torch.squeeze(x, dim=1)
        x = x.reshape(b, n_s, self.d_model)
        return x