import torch
import torch.nn as nn
import torch.nn.functional as F

def einsum_sum(A, B):
    """
    Perform the operation torch.sum(A * B, dim=-1) using torch.einsum.

    Args:
    A: Tensor of shape (n_batch, n_points, n_cluster, 3)
    B: Tensor of shape (n_batch, 1, n_cluster, 3)

    Returns:
    result: Tensor after summing the element-wise product along the last dimension.
    """
    # Use einsum to perform element-wise multiplication and sum along the last dimension
    result = torch.einsum('bpci,buci->bpc', A, B)
    return result

def reverse_indices(indices):
    reverse_indices_group = torch.argsort(torch.gather(torch.arange(indices.shape[1], device=indices.device).reshape(1, indices.shape[1]).expand_as(indices), dim=1, index=indices), dim=1)
    return reverse_indices_group

def compose_indices(indices_1, indices_2):
    new_indices = torch.gather(indices_1, dim=1, index=indices_2)
    return new_indices

def random_serialization(x: torch.Tensor, mask: torch.Tensor, n_clusters: int = 225):
    # x: input point cloud tensor(b, n, 3)
    # mask: input point cloud mask (b, n)
    # n_cluster: number of cluster ()
    with torch.no_grad():
        b, n, _ = x.shape
        random_pos = torch.randn(b, n, 1, device=x.device) + (1 - mask.unsqueeze(-1)) * 1e9
        random_indices = torch.argsort(random_pos, dim=1)[:, :n_clusters, :]
        random_selected_anchors = torch.take_along_dim(x, dim=1, indices=random_indices)

        x = x.unsqueeze(2)
        random_selected_anchors = random_selected_anchors.unsqueeze(1)
        x_subtracted = x - random_selected_anchors
        x_subtracted = x_subtracted * mask.reshape(b, n, 1, 1)

        random_weights = torch.randn(b, 1, n_clusters, 3, device=x.device)
        x_score = torch.sign(einsum_sum(x_subtracted, random_weights))
        ratios = torch.minimum(torch.sum(F.relu(x_score), dim=1, keepdim=True), torch.sum(F.relu(x_score * -1), dim=1, keepdim=True))

        ratios_indices = torch.argsort(ratios, dim=2, descending=True)
        ratios = torch.take_along_dim(ratios, indices=ratios_indices, dim=2)
        x_score = torch.take_along_dim(x_score, indices=ratios_indices, dim=2)

        coeffs = 2 ** torch.arange(127 - n_clusters, 127, 1, dtype=torch.float32, device=x.device)
        coeffs = coeffs.unsqueeze(-1)
        x_score = (x_score @ coeffs).squeeze(-1)
        x_score = x_score + (1 - mask) * torch.max(x_score)

        indices = torch.argsort(x_score, dim=1)

        del x_score
        del coeffs
        del ratios
        del ratios_indices
        del random_weights
        del x_subtracted
        del random_selected_anchors
        del random_pos

        return indices

def add_remainder_token(x, n_local, mask):
    b, n, d = x.shape
    remainder = n % n_local
    if remainder != 0:
        n_added_tokens = n_local - remainder
        return F.pad(x, (0, 0, 0, remainder)), F.pad(mask, (0, remainder))
    return x, mask, remainder

def split_token(x, n_local, mask):
    b, n, d = x.shape
    return x.reshape(b * n_local, n // n_local, d)

def revert_split(x, n_local, mask):
    b, n, d = x.shape
    return x.reshape(b // n_local, n * n_local, d)