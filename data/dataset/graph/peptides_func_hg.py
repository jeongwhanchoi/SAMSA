import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import numpy as np
import random
from tqdm import tqdm

class Peptide_func_hg_dataset():
    def __init__(self, root, split, device='cuda'):
        self.dataset = torch_geometric.datasets.LRGBDataset(root, 'Peptides-func', split=split)
        self.dataset_ = []
        self.d_len = len(self.dataset)
        self.indx = [i for i in range(self.d_len)]
        self.device = device
        random.shuffle(self.indx)
        self.pre_transformed_dataset = []
        with torch.no_grad():
            for i in tqdm(range(len(self.dataset))):
                current_indx = i
                dpoint = self.dataset[current_indx].cuda()
                
                adjacency_matrix = torch_geometric.utils.to_dense_adj(dpoint.edge_index, 
                                                                    edge_attr=torch.cat([dpoint.edge_attr, 
                                                                                        torch.ones(dpoint.edge_attr.shape[0], 1)], dim=1), 
                                                                    max_num_nodes=dpoint.x.shape[0])
                node_features = dpoint.x
                node_features = F.one_hot(node_features, 17).reshape(-1, 17 * 9)

                number_of_edge_vertex = dpoint.x.shape[0] + dpoint.edge_index.shape[1]
                number_of_edge_vertex_feature = node_features.shape[1] + adjacency_matrix.shape[-1] + 2
                number_of_vertex = dpoint.x.shape[0]
                number_of_edge = dpoint.edge_index.shape[1]
                number_of_vertex_feature = node_features.shape[1]
                number_of_edge_feature = adjacency_matrix.shape[-1]
                edge_vertex_feature_matrix = torch.zeros(number_of_edge_vertex, number_of_edge_vertex_feature)
                edge_vertex_connection_matrix = torch.zeros(1, number_of_edge_vertex, number_of_edge_vertex, 2)
                for j in range(number_of_vertex):
                    edge_vertex_feature_matrix[j,:number_of_vertex_feature] = node_features[j]
                    edge_vertex_feature_matrix[j,-1] = 1.0
                for j in range(number_of_edge):
                    v1, v2 = dpoint.edge_index[0][j], dpoint.edge_index[1][j]
                    edge_vertex_feature_matrix[number_of_vertex+j,number_of_vertex_feature:number_of_vertex_feature+number_of_edge_feature] = adjacency_matrix[0,v1,v2,:]
                    edge_vertex_connection_matrix[0, number_of_vertex+j, v1, 0] = 1
                    edge_vertex_connection_matrix[0, number_of_vertex+j, v2, 0] = 1
                    edge_vertex_connection_matrix[0, v1, number_of_vertex+j, 1] = 1
                    edge_vertex_connection_matrix[0, v2, number_of_vertex+j, 1] = 1

                self.pre_transformed_dataset.append((node_features, adjacency_matrix, dpoint.y))


    def get_graph(self):
        if len(self.indx) == 0:
            self.indx = [i for i in range(self.d_len)]
            random.shuffle(self.indx)
        current_indx = self.indx.pop()
        node_features, adjacency_matrix, label = self.pre_transformed_dataset[current_indx]
        return node_features, adjacency_matrix, label
    
    def get_batch(self, batch_size):
        nfs = []
        ams = []
        lbs = []
        nl = []
        mx_nodes = 256

        for i in range(batch_size):
            nf, am, lb = self.get_graph()
            mx_nodes = max(mx_nodes, nf.shape[0])
            nfs.append(nf)
            ams.append(am)
            lbs.append(lb)
            nl.append(nf.shape[0])

        n_node_features = nfs[0].shape[-1]
        n_edge_features = ams[0].shape[-1]

        node_features = torch.zeros(batch_size, mx_nodes, n_node_features)
        adjacency_matrices = torch.zeros(batch_size, mx_nodes, mx_nodes, n_edge_features)
        mask = torch.ones(batch_size, mx_nodes)

        for i in range(batch_size):
            node_features[i,:nl[i],:] = nfs[i]
            adjacency_matrices[i,:nl[i],:nl[i],:] = ams[i]
            mask[i,:nl[i]] = 0.0

        return node_features, adjacency_matrices, torch.stack(lbs, 0), mask