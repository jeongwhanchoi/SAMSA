import torch
import torch.nn as nn
from layer.utils import Copy
import math

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class LearnableEmbedding(nn.Module):
    def __init__(self, n_seq, d, p_drop, initialization='sinusoid1d'):
        super().__init__()
        self.dropout = nn.Dropout(p=p_drop)
        if initialization == 'normal':
            self.params = nn.Parameter(torch.randn(d, n_seq, 1, requires_grad=True), requires_grad=True)
        elif initialization == 'sinusoid1d':
            mat = positionalencoding1d(d, n_seq).unsqueeze(-1)
            mat = mat.transpose(0, 1)
            self.params = nn.Parameter(mat, requires_grad=True)
        elif initialization == 'sinusoid2d':
            mat = positionalencoding2d(d, int(math.sqrt(n_seq)), int(math.sqrt(n_seq))).reshape(d, n_seq, 1)
            self.params = nn.Parameter(mat, requires_grad=True)
        else:
            raise ValueError

    def forward(self, x, mask=None):
        b = x.shape[0]
        # p = Copy.apply(self.params, b)
        p = torch.cat([self.params for _ in range(b)], dim=2)
        p = p.transpose(0, 2)
        x = x + self.dropout(p)
        return x