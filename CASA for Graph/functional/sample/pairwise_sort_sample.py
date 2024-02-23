import torch
from functional.sample.batched_index_select import batched_index_select
from functional.sample.shuffle import shuffle
from functional.sample.soft_choice import soft_choice

def select_k(x, x_score, criteria, k):
    criteria, indices = torch.sort(criteria, dim=1, descending=True)
    indices = indices.squeeze(-1)
    x = batched_index_select(x, dim=1, index=indices)
    x_chosen, x_score_chosen = x[:,:k,:], x_score[:,:k,:]
    x_notchosen, x_score_notchosen = x[:,k:,:], x_score[:,k:,:]
    return x_chosen, x_score_chosen, x_notchosen, x_score_notchosen, indices

def pairwise_sort_sample(x, x_score, relative_map, k): 
    b, ns, nd = x.shape
    x_top, x_score_top, x_bottom, x_score_bottom, indices_top = select_k(x, x_score, x_score, k)
    while x_bottom.shape[1] < k:
        x_bottom = torch.cat([x_bottom, x_bottom], dim=1)
        x_score_bottom = torch.cat([x_score_bottom, torch.ones_like(x_score_bottom) * -1e8], dim=1)
    x_bottom, x_score_bottom, _, _, indices_below = select_k(x_bottom, x_score_bottom, torch.randn_like(x_score_bottom), k)
    x_top, x_score_top = x_top.unsqueeze(2), x_score_top.unsqueeze(2)
    x_bottom, x_score_bottom = x_bottom.unsqueeze(2), x_score_bottom.unsqueeze(2)
    x, x_score = torch.cat([x_top, x_bottom], dim=2), torch.cat([x_score_top, x_score_bottom], dim=2)
    x_score, _, _ = soft_choice(x_score)
    x = torch.sum(x * x_score, dim=2)
    return x, x_top.squeeze(2), x_bottom.squeeze(2), x_score