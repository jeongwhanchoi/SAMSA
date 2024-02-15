from layer.attention import Attention, CompositeMultiHeadAttention, EuclideanMultiHeadAttention, ScaledDotMultiHeadAttention, CMultiHeadAttention, CustomTransformerEncoderLayer, EuclideanTransformerEncoderLayer, CompositeTransformerEncoderLayer, ScaledDotTransformerEncoderLayer
from layer.samsa import SAM, CompositeSAMMultiHeadAttention, EuclideanSAMMultiHeadAttention, ScaledDotSAMMultiHeadAttention, SAMMultiHeadAttention, CustomSAMTransformerEncoderLayer, EuclideanSAMTransformerEncoderLayer, CompositeSAMTransformerEncoderLayer, ScaledDotSAMTransformerEncoderLayer
from layer.resc import RESC
from layer.point_layer import RandomColor
from layer.sequence_layer import LearnableEmbedding
from layer.pairwise_perceptron import SoftplusPWP
from layer.leaky_attention import RSTL
from layer.sscl import SSCL
from layer.utils import GlobalNode
import torch
import torch.nn as nn
import random
# from layer.knn import LocalAttention

class RSTLBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int,
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 attention_type: str="dot", norm_first=False,
                 norm_type="LayerNorm", activation='drelu', n_layers=1):
        super().__init__()
        self.mlist = nn.ModuleList([
            RSTL(d_model, nhead,
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                 dim_feedforward, dropout,
                 attention_type, norm_first,
                 norm_type, activation) for i in range(n_layers)
        ])
        self.d_model = d_model
    
    def forward(self, x, position, mask=None):
         regularization = torch.zeros(1, device=x.device)
         for i in range(len(self.mlist)):
            x, r = self.mlist[i](x, position, mask)
            regularization = regularization + r
         return x, regularization
    
class RSTLRBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int,
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 attention_type: str="dot", norm_first=False,
                 norm_type="LayerNorm", activation='drelu', n_layers=4, n_repeat=12):
        super().__init__()
        self.mlist = nn.ModuleList([
            RSTL(d_model, nhead,
                 n_sampled_points_lb, n_sampled_points_ub, drop_point, drop_global_points,
                 dim_feedforward, dropout,
                 attention_type, norm_first,
                 norm_type, activation) for i in range(n_layers)
        ])
        self.d_model = d_model
        self.n_repeat = n_repeat
    
    def forward(self, x, position, mask=None):
         regularization = torch.zeros(1, device=x.device)
         for j in range(self.n_repeat):
            perm = [i for i in range(len(self.mlist))]
            random.shuffle(perm)
            for i in perm:
               x, r = self.mlist[i](x, position, mask)
               regularization = regularization + r
         return x, regularization

__all__ = ['Attention', 
           'CompositeMultiHeadAttention', 'EuclideanMultiHeadAttention', 'ScaledDotMultiHeadAttention', 'CMultiHeadAttention', 
           'CustomTransformerEncoderLayer', 'EuclideanTransformerEncoderLayer', 'CompositeTransformerEncoderLayer', 'ScaledDotTransformerEncoderLayer',
           'CompositeSAMMultiHeadAttention', 'EuclideanSAMMultiHeadAttention', 'ScaledDotSAMMultiHeadAttention', 'SAMMultiHeadAttention', 
           'CustomSAMTransformerEncoderLayer', 'EuclideanSAMTransformerEncoderLayer', 'CompositeSAMTransformerEncoderLayer', 'ScaledDotSAMTransformerEncoderLayer',
           'RandomColor',
           'LearnableEmbedding',
           'SoftplusPWP',
           'GlobalNode',
           'RSTL', 'SSCL', 'RSTLBlock']
        #    ,
        #    'LocalAttention']