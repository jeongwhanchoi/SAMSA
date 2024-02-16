import torch

def pairwise_distance(x, y):
    with torch.no_grad():
        x_norm = (x**2).sum(2).unsqueeze(2)
        y_t = torch.transpose(y, 1, 2)
        y_norm = (y**2).sum(2).unsqueeze(1)
        dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
        return dist