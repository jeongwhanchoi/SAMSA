from ..name import module_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphBridge(nn.Module):
    """
        Generate Randomized Absolute Positional Encoding for Graphs.
        Input: node_features (b, n_nodes, d_nodes), edge_features (b, n_edges, d_edges), connection (b, n_edges, 2)
    """
    def __init__(self, d_model, d_random_feature):
        super().__init__()
        self.d_model = d_model
        self.d_random_features = d_random_feature
        self.hadamard_coeffs_l = nn.Parameter(torch.ones(1, 1, self.d_random_features), requires_grad=True)
        self.hadamard_coeffs_r = nn.Parameter(torch.ones(1, 1, self.d_random_features), requires_grad=True)
        self.mlp_node = nn.Sequential(
            nn.LazyLinear(d_model * 4),
            nn.GELU(),
            nn.LazyLinear(d_model),
        )
        self.mlp_edge = nn.Sequential(
            nn.LazyLinear(d_model * 4),
            nn.GELU(),
            nn.LazyLinear(d_model),
        )
        self.mlp_node_pe = nn.Sequential(
            nn.LazyLinear(d_model * 4),
            nn.GELU(),
            nn.LazyLinear(d_random_feature * 2),
        )

    def forward(self, node_features, edge_features, mask_node, mask_edge, connection, **kwargs):
        b, n_nodes, _ = node_features.shape
        node_random_positional_l = torch.randn(b, n_nodes, self.d_random_features, device=node_features.device).unsqueeze(2) * F.softplus(self.hadamard_coeffs_l)
        node_random_positional_r = torch.randn(b, n_nodes, self.d_random_features, device=node_features.device).unsqueeze(2) * F.softplus(self.hadamard_coeffs_r)
        node_random_positional = torch.cat([node_random_positional_l, node_random_positional_r], dim=2)
        node_features_add_pe = self.mlp_node_pe(node_features).reshape(node_features.shape[0], node_features.shape[1], 2, self.d_random_features)
        node_random_positional = node_random_positional + node_features_add_pe
        connection = connection.unsqueeze(-1).expand(-1, -1, -1, self.d_random_features)
        edge_pe = torch.gather(F.pad(node_random_positional, (0, 0, 0, 0, 0, edge_features.shape[1])),
                               dim=1, 
                               index=connection)
        edge_pe = torch.subtract(edge_pe[:, :, 0, :], edge_pe[:, :, 1, :])
        node_features = self.mlp_node(torch.cat([node_features, 
                                                    node_random_positional.reshape(b, 
                                                                                   n_nodes, 
                                                                                   2 * self.d_random_features)], 
                                                                                   dim=2))
        edge_features = self.mlp_edge(torch.cat([edge_features, edge_pe], dim=2))
        
        return {
            'x': torch.cat([node_features, edge_features], dim=1),
            'mask': torch.cat([mask_node, mask_edge], dim=1),
            'n_nodes': n_nodes,
            'n_edges': edge_features.shape[1]
        }
    
class GetNode(nn.Module):
    def forward(self, x, mask, **kwargs):
        return {
            'x': x[kwargs['n_nodes']],
            'x_nodes_edges': x,
            'mask': mask[kwargs['n_nodes']],
            'mask_nodes_edges': mask
        }

module_name_dict['GRAPHBRIDGE'] = GraphBridge
module_name_dict['GETNODE'] = GetNode