from ..name import module_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math

from timm.models.layers import DropPath

from .sublayer import *

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class Sinusoid(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        self.settings = dict(locals())
        self.settings["name"] = "SINUSOID"
        for k, v in self.settings.items():
            self.settings[k] = str(v)
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, x, **kwargs):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """

        tensor = x

        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc + tensor

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device, dtype=tensor.dtype)
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return {'x': self.cached_penc + tensor}

module_name_dict["SINUSOID"] = Sinusoid

class BridgeEmbedding(nn.Module):
    def __init__(self, 
                 d_model: int,
                 **kwargs):
        """
            Brief description:
                
            I/O:
                I: tensor of tokens (batch size, n_tokens, n_input_dimension)
                O: tensor of tokens (batch size, n_tokens, n_output_dimension)
        """
        super().__init__()
        self.settings = dict(locals())
        self.settings["name"] = "HSTRANS"
        for k, v in self.settings.items():
            self.settings[k] = str(v)
        self.combine = nn.Linear(d_model * 2, d_model)

    def forward(self, x, mask=None, **kwargs):
        rand_pos_emb = torch.randn_like(x)
        x = self.combine(torch.cat([x, rand_pos_emb], dim=2))

        rand_pos_emb_shifted = F.pad(rand_pos_emb[:, 1:, :], pad=(0, 0, 0, 1))
        bridge = rand_pos_emb - rand_pos_emb
        x = torch.cat([x, bridge], dim=1)

        mask_shifted = F.pad(mask[:, 1:], pad=(0, 1))
        bridge_mask = torch.minimum(mask, mask_shifted)
        mask = torch.cat([mask, bridge_mask], dim=1)
        return {'x': x, 'mask': mask}

module_name_dict["BRIDGE"] = BridgeEmbedding