import torch

def pairwise_distances(x, y):
    '''
    Input: x is a BxNxd matrix
           y is an optional BxMxd matirx
    Output: dist is a BxNxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    with torch.no_grad():
        x_norm = (x**2).sum(2).unsqueeze(2)
        y_t = torch.transpose(y, 1, 2)
        y_norm = (y**2).sum(2).unsqueeze(1)

        dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
        return dist