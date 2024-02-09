import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalNode(nn.Module):
    def __init__(self, n_nodes, n_dims):
        super().__init__()
        self.p = nn.Parameter(torch.randn(1, n_nodes, n_dims), requires_grad=True)
        self.n_nodes = n_nodes
        self.n_dims = n_dims

    def forward(self, x, x_ref=None, mask=None):
        x_ = torch.zeros(x.shape[0], x.shape[1] + self.n_nodes, x.shape[2], device=x.device)
        x_[:,:x.shape[1],:] += x
        x_[:,x.shape[1]:,:] += self.p

        x_ref_ = torch.zeros(x_ref.shape[0], x_ref.shape[1] + self.n_nodes, x_ref.shape[2], device=x.device)
        x_ref_[:,:x.shape[1],:] += x_ref

        if mask is not None:
            mask_ = torch.zeros(mask.shape[0], mask.shape[1] + self.n_nodes, device=mask.device)
            mask_[:,:mask.shape[1]] += mask
            return x_, x_ref_, mask_
        return x_, x_ref_, None
