import torch
from functional.sample.batched_index_select import batched_index_select
from functional.sample.shuffle import shuffle
from functional.sample.soft_choice import soft_choice

def sort_sample(x, x_score, k): 
    b, ns, nd = x.shape
    x_score, indices = torch.sort(x_score, dim=1, descending=True)
    indices = indices.squeeze(-1)
    x = batched_index_select(x, dim=1, index=indices)
    x_top, x_score_top = x[:,:k,:], x_score[:,:k,:]
    x_bottom, x_score_bottom = x[:,k:,:], x_score[:,k:,:]
    while x_bottom.shape[1] < k:
        if x_bottom.shape[1] == 0:
            x_bottom, x_score_bottom = x_top, torch.ones_like(x_score_top) * -1e8
            break
        x_bottom = torch.cat([x_bottom, x_bottom], dim=1)
        x_score_bottom = torch.cat([x_score_bottom, torch.ones_like(x_score_bottom) * -1e8], dim=1)
    x_bottom = torch.cat([x_bottom, x_score_bottom], dim=2)
    x_bottom = shuffle(x_bottom, axis=1)
    x_bottom, x_score_bottom = x_bottom[:,:,:-1], x_bottom[:,:,-1].unsqueeze(-1)
    c = k
    x_bottom, x_score_bottom = x_bottom[:,:c,:].reshape(b, k, c // k, nd), x_score_bottom[:,:c,:].reshape(b, k, c // k, 1)
    x_top, x_score_top = x_top.unsqueeze(2), x_score_top.unsqueeze(2)
    x, x_score = torch.cat([x_top, x_bottom], dim=2), torch.cat([x_score_top, x_score_bottom], dim=2)
    x_score, _, _ = soft_choice(x_score)
    x = torch.sum(x * x_score, dim=2)
    return x, x_top.squeeze(2), x_bottom.squeeze(2), x_score