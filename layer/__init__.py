import torch
import torch.nn as nn
import torch.nn.functional as F
from layer.attention.attention_score import maxout, scaled_dot, lily
from layer.attention.probability_function import softmax, leaky_softmax, relu_prob, leaky_relu_prob, leaky_softplus_prob, softplus_prob
from layer.attention.relative import pairwise_concat, pairwise_distance, pairwise_distance_similarity, pairwise_field, pairwise_placeholder, pairwise_hadamard, pairwise_lily, pairwise_similarity
from layer.attention import TransformerEncoderLayer
from layer.head import ClassificationHead, RegressionHead, SegmentationHead

class TransformerEncoderBlock(nn.Module):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int,
                 n_sampled_points, drop_point, drop_att, drop_val, drop_ffn,
                 score_functions, relative_function, probability_function,
                 norm_type, norm_position,
                 n_layers):
        super().__init__()
        if norm_position == "prenorm":
            norm_position = True
        else:
            norm_position = False

        if score_functions == "maxout":
            score_functions = maxout
        elif score_functions == "lily":
            score_functions = lily
        elif score_functions == "dot":
            score_functions = scaled_dot
        else:
            raise NotImplementedError

        relative_functions = nn.ModuleList()

        for i in range(n_layers):
            if relative_function == "cat":
                relative_functions.append(torch.compile(pairwise_concat()))
            elif relative_function == "dist":
                relative_functions.append(torch.compile(pairwise_distance()))
            elif relative_function == "distsim":
                relative_functions.append(torch.compile(pairwise_distance_similarity()))
            elif relative_function == "field":
                relative_functions.append(torch.compile(pairwise_field()))
            elif relative_function == "hadamard":
                relative_functions.append(torch.compile(pairwise_hadamard()))
            elif relative_function == "plily":
                relative_functions.append(torch.compile(pairwise_lily()))
            elif relative_function == "sim":
                relative_functions.append(torch.compile(pairwise_similarity(nhead)))
            elif relative_function == "none":
                relative_functions.append(torch.compile(pairwise_placeholder()))
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
        elif probability_function == "leakysoftplus":
            probability_function = leaky_softplus_prob
        elif probability_function == "softplus":
            probability_function = softplus_prob
        else:
            raise NotImplementedError
        
        probability_function = torch.compile(probability_function)
        
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, 
                                                             nhead, 
                                                             n_sampled_points, 
                                                             drop_point,
                                                             score_functions, 
                                                             relative_functions[i], 
                                                             probability_function,
                                                             norm=norm_type,
                                                             norm_first=norm_position,
                                                             dim_feedforward=d_model*4,
                                                             drop_att=drop_att,
                                                             drop_val=drop_val,
                                                             drop_ffn=drop_ffn) for i in range(n_layers)])

    def forward(self, x, position, relative_map, mask=None):
        for i in range(len(self.layers)):
            x = self.layers[i](x, position, relative_map, mask)
        return x
    
__all__ = ['TransformerEncoderBlock']