import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import random

class Peptide_func_dataset():
    def __init__(self, root, split):
        self.dataset = torch_geometric.datasets.LRGBDataset(root, 'Peptides-func', split=split)
        self.d_len = len(self.dataset)
        self.indx = [i for i in range(self.d_len)]
        self.pe = torch_geometric.transforms.AddRandomWalkPE(24, 'rwpe')
        random.shuffle(self.indx)

    def get_graph(self):
        if len(self.indx) == 0:
            self.indx = [i for i in range(self.d_len)]
            random.shuffle(self.indx)
        current_indx = self.indx.pop()
        dpoint = self.dataset[current_indx]
        dpoint = self.pe(dpoint)
        adjacency_matrix = torch_geometric.utils.to_dense_adj(dpoint.edge_index, edge_attr=dpoint.edge_attr, max_num_nodes=dpoint.x.shape[0])
        node_features = dpoint.x
        node_features = F.one_hot(node_features, 17).reshape(-1, 17 * 9)
        node_features = torch.cat([node_features, dpoint.rwpe], dim=1)
        return node_features, adjacency_matrix, dpoint.y
    
    def get_batch(self, batch_size):
        nfs = []
        ams = []
        lbs = []
        nl = []
        mx_nodes = 0

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
        mask = torch.zeros(batch_size, mx_nodes)

        for i in range(batch_size):
            node_features[i,:nl[i],:] = nfs[i]
            adjacency_matrices[i,:nl[i],:nl[i],:] = ams[i]
            mask[i,:nl[i]] = 1.0

        return node_features, adjacency_matrices, torch.stack(lbs, 0), mask