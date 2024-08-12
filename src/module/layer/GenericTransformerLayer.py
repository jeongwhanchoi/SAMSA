from ..name import module_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath

class GenericTransformerEncoderLayer(nn.Module):
    def __init__(self, 
                 d_model: int,
                 d_attention: int,
                 d_feedforward: int,
                 p_dropout_model: float,
                 p_dropout_attention_map: float,
                 p_droppath: float,
                 nhead: int,
                 output_key: str = 'x'
                 ):
        """
            Brief description:
                Implementation of a Generic/Vanilla (Full-Attention) Transformer Encoder Layer
                Main differences compared to Vanilla Attention:
                -   Tunable dropout rate for different part of feature transformation 
                -   Number of dimensions of input tokens can be designed to be different to the one of output tokens
                -   Tunable number of dimensions used for attention computation

            Hyperparameters:
                d_model (int): the number of dimensions of output tokens;
                d_attention (int): the number of dimensions of input tokens; 
                                   this dicates the rank of matmul(Q^T, K) matrix in attention computation
                                   in vanilla transformer, this is set equal to the number of dimensions of input tokens
                                   in our transformer, this should be lower or equal to the number of dimensions of input tokens
                d_feedforward (int): the number of dimensions of feedforward layer in MLP submodule of transformers
                p_dropout_xxx (float): ranging from 0.0 to 1.0 as probability number, 
                                       ideally this should be set between 0.0 (no regularization) to 0.5 (maximum regularization)
                                       there is four position where we can put regularization:
                                            - model/token: after the multi-head-attention and ffn modules
                                            - attention: the feature of query and key vectors
                                            - attention map, randomly zeroize relations (after softmax) in attention map
                                            - feedforward, randomly zeroize features after feedforward transformation
                nhead (int): number of attention heads
                                    
            I/O:
                I: tensor of tokens (batch size, n_tokens, n_input_dimension)
                O: tensor of tokens (batch size, n_tokens, n_output_dimension) n_output_dimension is d_model
        """
        super().__init__()
        self.settings = dict(locals())
        self.settings["name"] = "GTRANS"
        for k, v in self.settings.items():
            self.settings[k] = str(v)

        self.d_model = d_model
        self.d_attention = d_attention
        self.d_model_head = d_model // nhead
        self.d_attention_head = d_attention // nhead
        self.nhead = nhead
        self.p_dropout_attention_map = p_dropout_attention_map

        self.linear_q = nn.Linear(d_model, d_attention)

        self.linear_k = nn.Linear(d_model, d_attention)

        self.linear_v = nn.Linear(d_model, d_model)

        self.linear_cat = nn.Linear(d_model, d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.GELU(approximate='tanh'),
            nn.Dropout(p_dropout_model),
            nn.Linear(d_feedforward, d_model),
            nn.Dropout(p_dropout_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if p_droppath == 0.0:
            self.droppath = nn.Identity()
        else:
            self.droppath = DropPath(p_droppath)

        self.output_key = output_key

        torch.nn.init.xavier_uniform_(self.linear_q.weight)
        torch.nn.init.xavier_uniform_(self.linear_k.weight)
        torch.nn.init.xavier_uniform_(self.linear_v.weight)

    def forward(self, x, **kwargs):
        x_norm = self.norm1(x)
        q, k, v= self.linear_q(x_norm), self.linear_k(x_norm), self.linear_v(x_norm)
        q = q.view(x.shape[0], x.shape[1], self.nhead, self.d_attention_head).transpose(1, 2)
        k = k.view(x.shape[0], x.shape[1], self.nhead, self.d_attention_head).transpose(1, 2)
        v = v.view(x.shape[0], x.shape[1], self.nhead, self.d_model_head).transpose(1, 2)

        x_att = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p_dropout_attention_map)
        x_att = x_att.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.d_model)
        x = x + self.droppath(self.linear_cat(x_att)) * kwargs['residual_scale']
        x = x + self.droppath(self.mlp(self.norm2(x))) * kwargs['residual_scale']
        return {self.output_key: x}
    
module_name_dict["GTRANS"] = GenericTransformerEncoderLayer