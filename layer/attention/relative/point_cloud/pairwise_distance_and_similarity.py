import torch
import torch.nn as nn
from layer.attention.relative.point_cloud.pairwise_distance import pairwise_distance

class pairwise_distance_similarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        with torch.no_grad():
            x_coords, y_coords = x[:,:,:3], y[:,:,:3]
            x_surface, y_surface = x[:,:,3:6], y[:,:,3:6]
            dmap = pairwise_distance(x_coords, y_coords)
            smap = torch.bmm(x_surface, y_surface.transpose(1, 2))
            smap = smap.unsqueeze(-1)
            # while True:
            #     print(dmap.shape, smap.shape)
            return torch.cat([dmap, smap], dim=-1)