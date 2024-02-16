import torch
from layer.attention.relative.pairwise_distance import pairwise_distance

def pairwise_distance_similarity(x, y):
    with torch.no_grad():
        x_coords, y_coords = x[:,:,:3], y[:,:,:3]
        x_surface, y_surface = x[:,:,3:6], y[:,:,3:6]
        dmap = pairwise_distance(x_coords, y_coords)
        smap = torch.bmm(x_surface, y_surface.transpose(1, 2))
        return torch.cat([dmap, smap], dim=-1)