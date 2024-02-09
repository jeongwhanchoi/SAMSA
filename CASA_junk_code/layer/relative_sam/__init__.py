import torch
import torch.nn as nn
from layer.relative_sam.relative_sam_transformer_layer import RelativeScaledDotSAMTransformerEncoderLayer, RelativeEuclideanSAMTransformerEncoderLayer, RelativeCompositeSAMTransformerEncoderLayer

tel_type = {"dot": RelativeScaledDotSAMTransformerEncoderLayer,
            "euclid": RelativeEuclideanSAMTransformerEncoderLayer,
            "composite": RelativeCompositeSAMTransformerEncoderLayer}

class RSTL(nn.Module):
    def __init__(self, d_model: int, nhead: int,
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 attention_type: str="dot", norm_first=False,
                 norm_type="LayerNorm", activation='drelu'):
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
                                            normtype=norm_type,
                                            activation=activation)
        self.d_model = d_model
    
    def forward(self, x, position, mask=None):
        return self.layer(x, position, mask)

__all__ = ['RSTL']