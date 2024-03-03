import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.attention.attention_score import maxout, scaled_dot
from layer.attention.probability_function import softmax, leaky_softmax, relu_prob, leaky_relu_prob
from layer.attention.relative import pairwise_concat, pairwise_distance, pairwise_distance_similarity, pairwise_field
from layer.attention import TransformerEncoderLayer
from layer.head import ClassificationHead, RegressionHead, SegmentationHead

class TransformerEncoderBlock(nn.Module):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int,
                 n_sampled_points, drop_point,
                 score_functions, relative_function, probability_function,
                 n_layers):
        super().__init__()
        if score_functions == "maxout":
            score_functions = maxout
        elif score_functions == "dot":
            score_functions = scaled_dot
        else:
            raise NotImplementedError

        if relative_function == "cat":
            relative_function = pairwise_concat()
        elif relative_function == "dist":
            relative_function = pairwise_distance()
        elif relative_function == "distsim":
            relative_function = pairwise_distance_similarity()
        elif relative_function == "field":
            relative_function = pairwise_field()
        else:
            raise NotImplementedError

        if probability_function == "softmax":
            probability_function = softmax
        elif probability_function == "leakysoftmax":
            probability_function = leaky_softmax
        elif probability_function == "relu":
            probability_function = relu_prob
        elif probability_function == "leakyrelu":
            probability_function = leaky_relu_prob
        else:
            raise NotImplementedError
        
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, 
                                                             nhead, 
                                                             n_sampled_points, 
                                                             drop_point,
                                                             score_functions, 
                                                             relative_function, 
                                                             probability_function,
                                                             dim_feedforward=d_model*4) for _ in range(n_layers)])

    def forward(self, x, position, relative_map, mask=None):
        for i in range(len(self.layers)):
            x = self.layers[i](x, position, relative_map, mask)
        return x
    
__all__ = ['TransformerEncoderBlock']