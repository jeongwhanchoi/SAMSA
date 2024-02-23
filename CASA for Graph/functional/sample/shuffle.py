import torch

def shuffle(tensor, axis):
    row_perm = torch.rand(tensor.shape[:axis+1], device=tensor.device).argsort(axis)  # get permutation indices
    print(row_perm)
    for _ in range(tensor.ndim-axis-1): row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(axis+1)], *(tensor.shape[axis+1:]))  # reformat this for the gather operation
    return tensor.gather(axis, row_perm)