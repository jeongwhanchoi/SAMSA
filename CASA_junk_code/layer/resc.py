import torch
import torch.nn as nn
from layer.samsa import ScaledDotSAMTransformerEncoderLayer, EuclideanSAMTransformerEncoderLayer, CompositeSAMTransformerEncoderLayer

tel_type = {"dot": ScaledDotSAMTransformerEncoderLayer,
            "euclid": EuclideanSAMTransformerEncoderLayer,
            "composite": CompositeSAMTransformerEncoderLayer}

class RESC(nn.Module):
    def __init__(self, d_model: int, nhead: int,
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, leaky_softmax,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 attention_type: str="composite", norm_first=False):
        super().__init__()
        self.layer = tel_type[attention_type](d_model,
                                              nhead,
                                              n_sampled_points_lb,
                                              n_sampled_points_ub,
                                              drop_point,
                                              leaky_softmax,
                                              dim_feedforward,
                                              dropout,
                                              norm_first=norm_first)
        
    def forward(self, x, mask=None):
        return self.layer(x, mask)[0]