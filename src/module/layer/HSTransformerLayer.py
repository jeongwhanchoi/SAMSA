from ..name import module_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

import math

from timm.models.layers import DropPath

from .sublayer import *

class SWHSTransformerLayer(nn.Module):
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
                 hard,
                 output_key: str = 'x',
                 **kwargs):
        """
            Brief description:
                
            I/O:
                I: tensor of tokens (batch size, n_tokens, n_input_dimension)
                O: tensor of tokens (batch size, n_tokens, n_output_dimension)
        """
        super().__init__()
        self.settings = dict(locals())
        self.settings["name"] = "SWHSTRANS"
        for k, v in self.settings.items():
            self.settings[k] = str(v)

        self.d_model = d_model
        self.d_attention = d_attention
        self.d_model_head = d_model // nhead
        self.d_attention_head = d_attention // nhead
        self.nhead = nhead
        self.p_dropout_attention_map = p_dropout_attention_map
        self.n_sampled_token = n_sampled_token

        self.linear_q = nn.Linear(d_model, d_attention)
        self.linear_kv = nn.Linear(d_model, d_attention + d_model)
        self.linear_cat = nn.Linear(d_model, d_model)

        self.hard = hard
        self.output_key = output_key

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.GELU(),
            nn.Dropout(p_dropout_model),
            nn.Linear(d_feedforward, d_model),
            nn.Dropout(p_dropout_model),
        )

        self.sampler = DSZRC(nhead, n_sampled_token, d_model, d_attention + d_model, hard=hard, temperature=temperature)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        if p_droppath == 0.0:
            self.droppath = nn.Identity()
        else:
            self.droppath = DropPath(p_droppath)

        torch.nn.init.kaiming_uniform_(self.linear_q.weight, nonlinearity='linear')
        torch.nn.init.kaiming_uniform_(self.linear_kv.weight, nonlinearity='linear')

    def forward(self, x, mask=None, **kwargs):
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], device=x.device)
        x_norm = self.norm1(x)
        q, kv = self.linear_q(x_norm), self.linear_kv(x_norm)
        q = q.view(x.shape[0], x.shape[1], self.nhead, self.d_attention_head).transpose(1, 2)
        kv = kv.view(x.shape[0], x.shape[1], self.nhead, self.d_attention_head + self.d_model_head).transpose(1, 2)
        k_, v_ = torch.split(kv, split_size_or_sections=[self.d_attention_head, self.d_model_head], dim=-1)
        kv = self.sampler(x, kv, mask)
        k, v = torch.split(kv, split_size_or_sections=[self.d_attention_head, self.d_model_head], dim=-1)

        if self.training is True:
            dr = self.p_dropout_attention_map
        else:
            dr = 0.0

        x_att = F.scaled_dot_product_attention(q, k, v, dropout_p=dr) + flash_attn_func(q.half(), k_.half(), v_.half(), window_size=(self.n_sampled_token, 0)).to(x.dtype)
        x_att = x_att.transpose(1, 2).reshape(x_norm.shape[0], x_norm.shape[1], self.d_model)
        x = x + self.droppath(self.linear_cat(x_att)) * kwargs['residual_scale']
        x = x + self.droppath(self.mlp(self.norm2(x))) * kwargs['residual_scale']
        return {self.output_key: x}

module_name_dict["SWHSTRANS"] = SWHSTransformerLayer

class HSTransformerLayer(nn.Module):
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
                 hard,
                 output_key: str = 'x',
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

        self.d_model = d_model
        self.d_attention = d_attention
        self.d_model_head = d_model // nhead
        self.d_attention_head = d_attention // nhead
        self.nhead = nhead
        self.p_dropout_attention_map = p_dropout_attention_map
        self.n_sampled_token = n_sampled_token

        self.linear_q = nn.Linear(d_model, d_attention)
        self.linear_kv = nn.Linear(d_model, d_attention + d_model)
        self.linear_cat = nn.Linear(d_model, d_model)

        self.hard = hard
        self.output_key = output_key

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.GELU(),
            nn.Dropout(p_dropout_model),
            nn.Linear(d_feedforward, d_model),
            nn.Dropout(p_dropout_model),
        )

        self.sampler = DSZRC(nhead, n_sampled_token, d_model, d_attention + d_model, hard=hard, temperature=temperature)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        if p_droppath == 0.0:
            self.droppath = nn.Identity()
        else:
            self.droppath = DropPath(p_droppath)

        torch.nn.init.kaiming_uniform_(self.linear_q.weight, nonlinearity='linear')
        torch.nn.init.kaiming_uniform_(self.linear_kv.weight, nonlinearity='linear')

    def forward(self, x, mask=None, **kwargs):
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1], device=x.device)
        x_norm = self.norm1(x)
        q, kv = self.linear_q(x_norm), self.linear_kv(x_norm)
        q = q.view(x.shape[0], x.shape[1], self.nhead, self.d_attention_head).transpose(1, 2)
        kv = kv.view(x.shape[0], x.shape[1], self.nhead, self.d_attention_head + self.d_model_head).transpose(1, 2)

        kv = self.sampler(x, kv, mask)
        k, v = torch.split(kv, split_size_or_sections=[self.d_attention_head, self.d_model_head], dim=-1)

        x_att = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p_dropout_attention_map)
        x_att = x_att.transpose(1, 2).reshape(x_norm.shape[0], x_norm.shape[1], self.d_model)
        x = x + self.droppath(self.linear_cat(x_att)) * kwargs['residual_scale']
        x = x + self.droppath(self.mlp(self.norm2(x))) * kwargs['residual_scale']
        return {self.output_key: x}

module_name_dict["HSTRANS"] = HSTransformerLayer