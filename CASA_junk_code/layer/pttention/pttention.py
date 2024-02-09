import torch
import torch.nn as nn
import torch.nn.functional as F

def sample(x, x_score, k, sampler):
    b, ns, nd = x.shape
    x_score, indices = torch.sort(x_score, dim=1, descending=True)
    indices = indices.squeeze(-1)
    x = batched_index_select(x, dim=1, index=indices)
    x_top, x_score_top = x[:,:k,:], x_score[:,:k,:]
    x_bottom, x_score_bottom = x[:,k:,:], x_score[:,k:,:]
    x_bottom = torch.cat([x_bottom, x_score_bottom], dim=2)
    x_bottom = shufflerow(x_bottom, axis=1)
    x_bottom, x_score_bottom = x_bottom[:,:,:-1], x_bottom[:,:,-1].unsqueeze(-1)
    c = k
    x_bottom, x_score_bottom = x_bottom[:,:c,:].reshape(b, k, c // k, nd), x_score_bottom[:,:c,:].reshape(b, k, c // k, 1)
    x_top, x_score_top = x_top.unsqueeze(2), x_score_top.unsqueeze(2)
    x, x_score = torch.cat([x_top, x_bottom], dim=2), torch.cat([x_score_top, x_score_bottom], dim=2)
    x_score = sampler(x_score, dim=2)
    x = torch.sum(x * x_score, dim=2)
    return x

def sample_test(x, x_score, k):
    b, ns, nd = x.shape
    x_score, indices = torch.sort(x_score, dim=1, descending=True)
    indices = indices.squeeze(-1)
    x = batched_index_select(x, dim=1, index=indices)
    x_top = x[:,:k,:]
    return x_top

class SAM_(nn.Module):
    def __init__(self, 
                 n_sampled_points_lb,
                 n_sampled_points_ub,
                 d_model,
                 drop_point: float = 0.1):
        super().__init__()
        self.n_sampled_points_lb = n_sampled_points_lb
        self.n_sampled_points_ub = n_sampled_points_ub
        self.d_model = d_model
        self.pointwise = nn.Sequential(nn.LayerNorm(d_model),
                                       nn.LazyLinear(1))
        self.drop_point = 1 - drop_point
        self.sample = LowTSoftmax()

    def forward(self, x, v=None, mask=None):
        x_fwd = self.pointwise(x) / math.sqrt(2)
        if mask is not None:
            mask = mask.unsqueeze(dim=2)
            x_fwd = x_fwd + mask * (-1e9)

        if self.training == True:
            n_sampled_points = random.randint(self.n_sampled_points_lb, self.n_sampled_points_ub)
            drop_ = torch.bernoulli(torch.ones(x.shape[0], x.shape[1], device=x.device) * self.drop_point).unsqueeze(dim=2)
            x_fwd = x_fwd + (1 -  drop_) * (-1e9)
            if v != None:
                x = torch.cat([x, v], dim=-1)
            return sample(x, x_fwd, n_sampled_points, self.sample)
        else:
            if v != None:
                x = torch.cat([x, v], dim=-1)
            return sample_test(x, x_fwd, self.n_sampled_points_ub)