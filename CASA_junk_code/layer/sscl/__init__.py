import torch
import torch.nn as nn
from layer.sscl.sscl_tl import SSCL_ScaledDotSAMTransformerEncoderLayer, SSCL_EuclideanSAMTransformerEncoderLayer, SSCL_CompositeSAMTransformerEncoderLayer

tel_type = {"dot": SSCL_ScaledDotSAMTransformerEncoderLayer,
            "euclid": SSCL_EuclideanSAMTransformerEncoderLayer,
            "composite": SSCL_CompositeSAMTransformerEncoderLayer}

class SSCL(nn.Module):
    def __init__(self, d_model: int, nhead: int,
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 attention_type: str="dot", norm_first=False,
                 norm_type="LayerNorm"):
        super().__init__()
        self.layer = tel_type[attention_type](d_model,
                                            nhead,
                                            n_sampled_points_lb,
                                            n_sampled_points_ub,
                                            drop_point,
                                            drop_global_points,
                                            dim_feedforward,
                                            dropout,
                                            norm_first=norm_first,
                                            normtype=norm_type)
        self.d_model = d_model
    
    def forward(self, x, position, mask=None):
        return self.layer(x, position, mask)

__all__ = ['RSTL']