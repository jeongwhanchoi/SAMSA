import torch

# def get_neighbors(points, idx):
#     rets = []
#     for i in range(idx.shape[2]):
#         z = torch.cat([idx[:, :, i].unsqueeze(-1) for _ in range(points.shape[2])], dim=-1)
#         rets.append(torch.gather(points, dim=1, index=z))
#     return torch.stack(rets, dim=2)

def get_neighbors(point, idx):
    new_indices = torch.cat([idx.reshape(idx.shape[0], idx.shape[1] * idx.shape[2], 1) for _ in range(point.shape[2])], dim=-1)
    ret = torch.gather(point, dim=1, index=new_indices)
    return ret.reshape(point.shape[0], point.shape[1], idx.shape[2], point.shape[2])