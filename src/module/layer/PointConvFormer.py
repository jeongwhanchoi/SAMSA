from ..name import module_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from timm.models.layers import DropPath

from .sublayer import *

class ConvRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial ConvRMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for ConvRMSNorm, disabled by
            default because ConvRMSNorm doesn't enforce re-centering invariance.
        """
        super(ConvRMSNorm, self).__init__()

        self.eps = eps
        self.d = d

    def forward(self, x):
        norm_x = x.norm(2, dim=1, keepdim=True)
        rms_x = norm_x * 1 / math.sqrt(self.d)
        x_normed = x / (rms_x + self.eps)

        return x_normed

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

def select_one_random_duplet(points, mask):
    with torch.no_grad():
        select_score = torch.rand(points.shape[0], points.shape[1], device=points.device) + (1 - mask) * 1e9
        select_indices = select_score.argmin(dim=1, keepdim=True)[:, :1]
        ret1 = torch.take_along_dim(points, indices=select_indices.unsqueeze(-1), dim=1)

        select_score = torch.rand(points.shape[0], points.shape[1], device=points.device) + (1 - mask) * 1e9
        select_indices = select_score.argmin(dim=1, keepdim=True)[:, :1]
        ret2 = torch.take_along_dim(points, indices=select_indices.unsqueeze(-1), dim=1)

        ret = (ret1 + ret2) / 2.0

        del select_score
        del select_indices
        del ret1
        del ret2
        return ret

def weighted_mean(points, mask):
    with torch.no_grad():
        n_points = torch.sum(mask, dim=1, keepdim=True).unsqueeze(-1)
        ret = torch.sum(points * mask.unsqueeze(-1), dim=1, keepdim=True) / (n_points + 1e-9)
        del n_points
        return ret

def align_points(points, mask_l, mask_r):
    with torch.no_grad():
        points_l = points - select_one_random_duplet(points, mask_l)
        points_r = points - select_one_random_duplet(points, mask_r)

        ret = points_l * mask_l.unsqueeze(-1) + points_r * mask_r.unsqueeze(-1)

        del points_l
        del points_r
        return ret

def random_serialization(points, mask, n_depth):
    # points: Tensor(B, N, 3)
    # mask: Tensor(B, N), 1 = there's a point, 0 = there's no point
    # n_depth: int, how many times to divides the space
    with torch.no_grad():
        inf = 2 ** (n_depth // 2 + 3)
        score_scale = 2 ** (n_depth // 2 + 1)
        total_score = torch.zeros(points.shape[0], points.shape[1], 1, device=points.device)
        oldABC = torch.ones(points.shape[0], 3, 1, device=points.device)

        for i in range(n_depth):
            ABC = torch.randn(points.shape[0], 3, 1, device=points.device)

            angle = torch.sum(F.normalize(ABC, dim=1) * F.normalize(oldABC, dim=1), dim=1, keepdim=True)
            signABC = torch.sign(angle)
            signABC = torch.where(signABC == 0, 1.0, signABC)
            ABC = ABC * signABC
            oldABC = ABC

            projection = points @ ABC
            D = (weighted_mean(projection, mask) * -1)
            projection = projection + D
            score = torch.sign(projection)

            total_score += score * score_scale
            score_scale = score_scale // 2

            mask_l = torch.minimum(F.relu(score * -1).squeeze(2), mask)
            mask_r = torch.minimum(F.relu(score).squeeze(2), mask)

            points = align_points(points, mask_l, mask_r)

            del mask_l
            del mask_r
            del score
            del projection
            del D
            del ABC
            del signABC

        total_score = total_score + inf * (1 - mask).unsqueeze(-1)
        indices = torch.argsort(total_score, dim=1)

        return indices.squeeze(-1)

class Serialization(nn.Module):
    def __init__(self, serialization_depth):
        super().__init__()
        self.serialization_depth = serialization_depth

    def forward(self, x, x_coords, mask, **kwargs):
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], device=x.device)
        serialization_indices = random_serialization(x_coords, mask, self.serialization_depth)
        reverse_serialization_indices = reverse_indices(serialization_indices)
        x = torch.take_along_dim(x, indices=serialization_indices.unsqueeze(2), dim=1)
        return {'x': x, 'reverse_serialization_indices': reverse_serialization_indices}

class ReverseSerialization(nn.Module):
    def forward(self, x, reverse_serialization_indices, **kwargs):
        x = torch.take_along_dim(x, indices=reverse_serialization_indices.unsqueeze(2), dim=1)
        return {'x': x}

module_name_dict["SERIALIZE"] = Serialization
module_name_dict["RSERIALIZE"] = ReverseSerialization


def add_remainder_token(x, n_local, mask):
    b, d, n = x.shape
    remainder = n_local - (n - n // n_local * n_local)
    if remainder != 0:
        n_added_tokens = n_local - remainder
        return F.pad(x, (0, remainder)), F.pad(mask, (0, remainder)), remainder
    return x, mask, remainder

def add_remainder_token_trans(x, n_local, mask):
    b, n, d = x.shape
    remainder = n_local - (n - n // n_local * n_local)
    if remainder != 0:
        n_added_tokens = n_local - remainder
        return F.pad(x, (0, 0, 0, remainder)), F.pad(mask, (0, remainder)), remainder
    return x, mask, remainder

def downsample(x, mask, group_size=2):
    x, mask, remainder = add_remainder_token(x, group_size, mask)
    b, d, n = x.shape
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2] // group_size, group_size)
    with torch.no_grad():
        mask = mask.reshape(mask.shape[0], 1, (mask.shape[1] // group_size), group_size)
        mask_multiplier = torch.sum(mask, dim=3, keepdim=True)
        mask_multiplier = (1 / torch.where(mask_multiplier == 0, 100, mask_multiplier)) * mask
    x = torch.sum(x * mask_multiplier, dim=3, keepdim=True) 
    mask, _ = torch.max(mask, dim=3, keepdim=True)
    x, mask = x.reshape(b, d, n // group_size), mask.reshape(b, n // group_size)
    return x, mask, remainder

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, **kwargs):
        super().__init__()
        pad = kernel_size // 2
        self.layer = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, groups=groups, **kwargs)

    def forward(self, x, **kwargs):
        return self.layer(x)[..., :x.shape[-1]]

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim_feedforward, kernel_size, droppath=0.3):
        super().__init__()
        self.conv1 = Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size, 
            groups=in_channels)
        self.norm = ConvRMSNorm(in_channels)
        self.conv2 = Conv1d(
            in_channels=in_channels, 
            out_channels=dim_feedforward,
            kernel_size=1, 
            groups=1)

        self.b1 = nn.Sequential(
            self.conv1, 
            self.norm,
            self.conv2
        )

        self.act = nn.GELU()

        self.select_main_stream = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=in_channels,
            kernel_size=1, 
            groups=1
        )
        self.select_b_stream = nn.Conv1d(
            in_channels=dim_feedforward, 
            out_channels=in_channels,
            kernel_size=1, 
            groups=1
        )
        self.conv3 = Conv1d(
            in_channels=dim_feedforward, 
            out_channels=out_channels,
            kernel_size=1, 
            groups=1)

        if in_channels != out_channels:
            self.residual_transf = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1, 
                groups=1
            )
        else:
            self.residual_transf = nn.Identity()

        if droppath == 0.0:
            self.droppath = nn.Identity()
        else:
            self.drop_path = DropPath(droppath)

    def forward(self, x, mask, **kwargs):
        x_branch = self.b1(x)

        x_sm = self.select_main_stream(x)
        x_sb = self.select_b_stream(x_branch)
        x_att = F.sigmoid(torch.mean(x_sm * x_sb, dim=1, keepdim=True))

        x_branch = self.conv3(self.act(x_branch))
        
        return self.residual_transf(x) + self.drop_path(x_branch) * x_att * kwargs['residual_scale']

class ConvTranspose1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.layer = nn.ConvTranspose1d(*args, **kwargs)
    
    def forward(self, x, *args, **kwargs):
        return self.layer(x)

class SerializedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, dim_feedforward, kernel_size, droppath=0.3, serialization_depth=32, conv_type='causal', output_key: str = 'x',):
        super().__init__()

        self.settings = dict(locals())
        self.settings["name"] = "SERCONV"
        for k, v in self.settings.items():
            self.settings[k] = str(v)

        if conv_type == 'masked':
            conv_block_type = Conv1dBlock
        else:
            raise ValueError

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim_feedforward = dim_feedforward
        self.kernel_size = kernel_size
        self.conv_type = conv_type
        self.droppath = droppath
        self.serialization_depth = serialization_depth
        self.output_key = output_key

        self.conv = conv_block_type(in_channels, out_channels, dim_feedforward, kernel_size, droppath)
    
    def forward(self, x, x_coords, mask=None, **kwargs):
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], device=x.device)
        serialization_indices = random_serialization(x_coords, mask, self.serialization_depth)
        reverse_serialization_indices = reverse_indices(serialization_indices)
        x = torch.take_along_dim(x, indices=serialization_indices.unsqueeze(2), dim=1)
        x = x.transpose(1, 2)
        if self.conv_type == 'masked':
            x = x * mask.unsqueeze(1)
        x = self.conv(x, residual_scale=kwargs['residual_scale'], mask=mask)
        x = torch.take_along_dim(x, indices=reverse_serialization_indices.unsqueeze(1), dim=2)
        x = x.transpose(1, 2)
        return {self.output_key: x}

module_name_dict["SERCONV"] = SerializedConvolution

class SerializedLHSTransformer(nn.Module):
    def __init__(self, 
                 d_model: int,
                 d_attention: int,
                 d_feedforward: int,
                 p_dropout_model: float,
                 p_dropout_attention_map: float,
                 p_droppath: float,
                 nhead: int,
                 n_sampled_token: int,
                 temperature: float,
                 serialization_depth: int=32,
                 n_local_layers: int=4,
                 output_key: str = 'x',
                 **kwargs
                 ):

        super().__init__()

        self.settings = dict(locals())
        self.settings["name"] = "SERLHSTRAN"
        for k, v in self.settings.items():
            self.settings[k] = str(v)

        self.d_model = d_model
        self.d_attention = d_attention
        self.d_model_head = d_model // nhead
        self.d_attention_head = d_attention // nhead
        self.nhead = nhead
        self.p_dropout_attention_map = p_dropout_attention_map
        self.n_sampled_token = n_sampled_token
        self.temperature = temperature
        self.serialization_depth = serialization_depth
        self.n_local_layers = n_local_layers

        if p_droppath == 0.0:
            self.droppath = nn.Identity()
        else:
            self.droppath = DropPath(p_droppath)

        self.linear_q = nn.ModuleList([nn.Linear(d_model, d_attention) for _ in range(n_local_layers + 1)])
        self.linear_kv = nn.ModuleList([nn.Linear(d_model, d_attention + d_model) for _ in range(n_local_layers + 1)])
        self.linear_cat = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_local_layers + 1)])

        self.mlp = nn.ModuleList([nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.GELU(),
            nn.Dropout(p_dropout_model),
            nn.Linear(d_feedforward, d_model),
            nn.Dropout(p_dropout_model),
        ) for _ in range(n_local_layers + 1)])

        self.sampler = nn.ModuleList([DSZRC(nhead, n_sampled_token, d_model, d_attention + d_model, temperature) for _ in range(n_local_layers + 1)])

        self.norm1 = nn.ModuleList([RMSNorm(d_model) for _ in range(n_local_layers + 1)])
        self.norm2 = nn.ModuleList([RMSNorm(d_model) for _ in range(n_local_layers + 1)])
        self.output_key = output_key

    def forward(self, x, x_coords, mask=None, **kwargs):
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], device=x.device)
        serialization_indices = random_serialization(x_coords, mask, self.serialization_depth)
        reverse_serialization_indices = reverse_indices(serialization_indices)
        x = torch.take_along_dim(x, indices=serialization_indices.unsqueeze(2), dim=1)
        orig_b, orig_n, orig_d = x.shape
        x, mask, remainder = add_remainder_token_trans(x, self.n_sampled_token, mask)

        x = x.reshape(x.shape[0] * self.n_sampled_token, x.shape[1] // self.n_sampled_token, x.shape[2])
        mask = mask.reshape(mask.shape[0] * self.n_sampled_token, mask.shape[1] // self.n_sampled_token)

        for i in range(self.n_local_layers):
            x_norm = self.norm1[i](x)
            q, kv = self.linear_q[i](x_norm), self.linear_kv[i](x_norm)
            q = q.view(x.shape[0], x.shape[1], self.nhead, self.d_attention_head).transpose(1, 2)
            kv = kv.view(x.shape[0], x.shape[1], self.nhead, self.d_attention_head + self.d_model_head).transpose(1, 2)

            kv = self.sampler[i](x_norm, kv, mask)
            k, v = torch.split(kv, split_size_or_sections=[self.d_attention_head, self.d_model_head], dim=-1)

            x_att = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p_dropout_attention_map)
            x_att = x_att.transpose(1, 2).reshape(x_norm.shape[0], x_norm.shape[1], self.d_model)
            x = x + self.droppath(self.linear_cat[i](x_att)) * kwargs['residual_scale']
            x = x + self.droppath(self.mlp[i](self.norm2[i](x))) * kwargs['residual_scale']

        x = x.reshape(x.shape[0] // self.n_sampled_token, x.shape[1] * self.n_sampled_token, -1)
        x = x[:, :orig_n, :]
        mask = mask.reshape(mask.shape[0] // self.n_sampled_token, mask.shape[1] * self.n_sampled_token)
        mask = mask[:, :orig_n]

        i = -1
        x_norm = self.norm1[i](x)
        q, kv = self.linear_q[i](x_norm), self.linear_kv[i](x_norm)
        q = q.view(x.shape[0], x.shape[1], self.nhead, self.d_attention_head).transpose(1, 2)
        kv = kv.view(x.shape[0], x.shape[1], self.nhead, self.d_attention_head + self.d_model_head).transpose(1, 2)

        kv = self.sampler[i](x_norm, kv, mask)
        k, v = torch.split(kv, split_size_or_sections=[self.d_attention_head, self.d_model_head], dim=-1)

        x_att = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p_dropout_attention_map)
        x_att = x_att.transpose(1, 2).reshape(x_norm.shape[0], x_norm.shape[1], self.d_model)
        x = x + self.droppath(self.linear_cat[i](x_att)) * kwargs['residual_scale']
        x = x + self.droppath(self.mlp[i](self.norm2[i](x))) * kwargs['residual_scale']

        x = torch.take_along_dim(x, indices=reverse_serialization_indices.unsqueeze(-1), dim=1)

        return {self.output_key: x}

module_name_dict["SERLHSTRAN"] = SerializedLHSTransformer