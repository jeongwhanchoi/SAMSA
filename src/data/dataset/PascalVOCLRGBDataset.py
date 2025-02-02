import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import random
from tqdm import tqdm

def preprocess_graph(graph, max_n_nodes, max_n_edges):
    x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
    node_features = torch.zeros(1, max_n_nodes, x.shape[1])
    node_features[:, :x.shape[0], :] = x.unsqueeze(0)
    mask_node = torch.ones(1, max_n_nodes, 1)
    mask_node[:, x.shape[0]:, :] = 0.0

    connection = torch.ones(1, max_n_edges, 2, dtype=torch.long) * (max_n_edges - 1)
    connection[:, :edge_index.shape[1], :] = edge_index.transpose(0, 1).unsqueeze(0)

    edge_features = torch.zeros(1, max_n_edges, edge_attr.shape[1])
    edge_features[:, :edge_attr.shape[0], :] = edge_attr.unsqueeze(0)
    mask_edge = torch.ones(1, max_n_edges, 1)
    mask_edge[:, edge_attr.shape[0]:, :] = 0.0

    return node_features, edge_features, mask_node, mask_edge, connection

class Pascalvoc_dataset():
    def __init__(self, root, split, pe='None', device='cuda'):
        self.dataset = torch_geometric.datasets.LRGBDataset(root, 'pascalvoc-sp', split=split)
        self.dataset_ = []
        self.device = device
        self.d_len = len(self.dataset)
        self.indx = [i for i in range(self.d_len)]
        random.shuffle(self.indx)
        self.pre_transformed_dataset = []
        self.max_n_nodes = 0
        self.max_n_edges = 0
        if pe == 'rwpe':
            PE_gen = torch_geometric.transforms.AddRandomWalkPE(walk_length=24, attr_name='PE')

        with torch.no_grad():
            for i in tqdm(range(len(self.dataset))):
                self.max_n_nodes = max(self.max_n_nodes, dpoint.x.shape[0])
                self.max_n_edges = max(self.max_n_edges, dpoint.edge_attr.shape[0])


            for i in tqdm(range(len(self.dataset))):
                current_indx = i
                dpoint = self.dataset[current_indx]
                adjacency_matrix = torch_geometric.utils.to_dense_adj(dpoint.edge_index, 
                                                                    edge_attr=torch.cat([dpoint.edge_attr, 
                                                                                        torch.ones(dpoint.edge_attr.shape[0], 1)], dim=1), 
                                                                    max_num_nodes=dpoint.x.shape[0])
                node_features = dpoint.x.to(self.device)
                if pe != "None":
                    pe_feature = PE_gen(dpoint.to(self.device))['PE']
                    node_features = torch.cat([node_features, pe_feature], dim=-1)
                self.pre_transformed_dataset.append((node_features, adjacency_matrix, dpoint.y))
                self.max_n_nodes = max(self.max_n_nodes, node_features.shape[0])
                

    def get_graph(self):
        if len(self.indx) == 0:
            self.indx = [i for i in range(self.d_len)]
            random.shuffle(self.indx)
        current_indx = self.indx.pop()
        node_features, adjacency_matrix, label = self.pre_transformed_dataset[current_indx]
        return node_features.to(self.device), adjacency_matrix.to(self.device), label.to(self.device)
    
    def get_batch(self, batch_size):
        nfs = []
        ams = []
        lbs = []
        nl = []
        mx_nodes = self.max_n_nodes

        for i in range(batch_size):
            nf, am, lb = self.get_graph()
            mx_nodes = max(mx_nodes, nf.shape[0])
            nfs.append(nf)
            ams.append(am)
            lbs.append(lb.reshape(-1))
            nl.append(nf.shape[0])

        n_node_features = nfs[0].shape[-1]
        n_edge_features = ams[0].shape[-1]

        node_features = torch.zeros(batch_size, mx_nodes, n_node_features).to(self.device)
        adjacency_matrices = torch.zeros(batch_size, mx_nodes, mx_nodes, n_edge_features).to(self.device)
        mask = torch.ones(batch_size, mx_nodes).to(self.device)

        for i in range(batch_size):
            node_features[i,:nl[i],:] = nfs[i]
            adjacency_matrices[i,:nl[i],:nl[i],:] = ams[i]
            mask[i,:nl[i]] = 0.0
        return node_features, adjacency_matrices, torch.cat(lbs, 0), mask, nl