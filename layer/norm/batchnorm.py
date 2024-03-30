import torch
import torch.nn as nn
from torch.nn import init


class MaskedBatchNorm1d(nn.Module):
    """ A masked version of nn.BatchNorm1d. Only tested for 3D inputs.
        Args:
            num_features: :math:`C` from an expected input of size
                :math:`(N, L, C)`
            eps: a value added to the denominator for numerical stability.
                Default: 1e-5
            momentum: the value used for the running_mean and running_var
                computation. Can be set to ``None`` for cumulative moving average
                (i.e. simple average). Default: 0.1
            affine: a boolean value that when set to ``True``, this module has
                learnable affine parameters. Default: ``True``
            track_running_stats: a boolean value that when set to ``True``, this
                module tracks the running mean and variance, and when set to ``False``,
                this module does not track such statistics and always uses batch
                statistics in both training and eval modes. Default: ``True``
        Shape:
            - Input: :math:`(N, L, C)`
            - input_mask: (N, 1, L) tensor of ones and zeros, where the zeros indicate locations not to use.
            - Output: :math:`(N, C)` or :math:`(N, L, C)` (same shape as input)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(MaskedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = nn.Parameter(torch.tensor(1.0))
        self.momentum_ = nn.Parameter(torch.tensor(momentum))
        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.register_buffer('running_var', torch.ones(num_features, 1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, input, input_mask=None):
        # Calculate the masked mean and variance
        B, C, L = input.shape
        input = input.transpose(1, 2)
        with torch.no_grad():
            input_mask = input_mask.unsqueeze(1)
            input_mask = 1 - input_mask
            masked = input * input_mask
            n = input_mask.sum()
            # Sum
            masked_sum = masked.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True)
            # Divide by sum of mask
            current_mean = masked_sum / n
            current_var = ((masked - current_mean) ** 2).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True) / n
            # Update running stats
            if self.training:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * current_var
                self.momentum = self.momentum_
        # Norm the input
        normed = (masked - self.running_mean) / (torch.sqrt(self.running_var + self.eps))
        normed = normed.transpose(1, 2)
        return normed