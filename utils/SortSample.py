import torch
from .BatchedIndexSelect import BatchedIndexSelect
from .Shuffle import Shuffle
from .SoftChoice import SoftChoice

def SortSample(x, x_score, k): 
    b, ns, nd = x.shape
    x_score, indices = torch.sort(x_score, dim=1, descending=True)
    indices = indices.squeeze(-1)
    x = BatchedIndexSelect(x, dim=1, index=indices)
    x_top, x_score_top = x[:,:k,:], x_score[:,:k,:]
    x_bottom, x_score_bottom = x[:,k:,:], x_score[:,k:,:]
    x_bottom = torch.cat([x_bottom, x_score_bottom], dim=2)
    x_bottom = Shuffle(x_bottom, axis=1)
    x_bottom, x_score_bottom = x_bottom[:,:,:-1], x_bottom[:,:,-1].unsqueeze(-1)
    c = k
    x_bottom, x_score_bottom = x_bottom[:,:c,:].reshape(b, k, c // k, nd), x_score_bottom[:,:c,:].reshape(b, k, c // k, 1)
    x_top, x_score_top = x_top.unsqueeze(2), x_score_top.unsqueeze(2)
    x, x_score = torch.cat([x_top, x_bottom], dim=2), torch.cat([x_score_top, x_score_bottom], dim=2)
    x_score, _, _ = SoftChoice(x_score, dim=2)
    x = torch.sum(x * x_score, dim=2)
    return x, x_top, x_bottom, x_score