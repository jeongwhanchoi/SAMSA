import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.attention.attention_score import maxout, scaled_dot
from layer.attention.probability_function import softmax, leaky_softmax, relu_prob, leaky_relu_prob
from layer.attention.relative import pairwise_concat, pairwise_distance, pairwise_distance_similarity
from layer.attention import TransformerEncoderLayer
from layer.pc_layer import RandomColor
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
            relative_function = pairwise_concat
        elif relative_function == "dist":
            relative_function = pairwise_distance
        elif relative_function == "distsim":
            relative_function = pairwise_distance_similarity
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
                                                             probability_function) for _ in range(n_layers)])
        
        self.n_layers = n_layers
        self.d_model = d_model
        self.learnable_layer_skip = nn.Parameter(torch.zeros(1, 1, 1, self.n_layers, self.n_layers), requires_grad=True)

    def forward(self, x, position, mask=None):
        x_skip = torch.zeros(x.shape[0],
                             x.shape[1], 
                             self.d_model, 
                             self.n_layers,
                             device=x.device)
        
        for i in range(len(self.layers)):
            x = self.layers[i](x, position, mask) + x_skip[:,:,:,i]
            x_skip = x_skip + x.unsqueeze(-1) * self.learnable_layer_skip[:,:,:,i,:]
        return x
    
__all__ = ['TransformerEncoderBlock', 
           'RandomColor',
           'ClassificationHead', 'RegressionHead', 'SegmentationHead']