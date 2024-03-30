import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import random
from tqdm import tqdm

class Peptide_func_dataset():
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
                dpoint = self.dataset[current_indx]
                adjacency_matrix = torch_geometric.utils.to_dense_adj(dpoint.edge_index, 
                                                                    edge_attr=torch.cat([dpoint.edge_attr, 
                                                                                        torch.ones(dpoint.edge_attr.shape[0], 1)], dim=1), 
                                                                    max_num_nodes=dpoint.x.shape[0])
                # adjacency_matrix = F.one_hot(adjacency_matrix.long(), num_classes=4).reshape(1,
                #                                                                       adjacency_matrix.shape[1],
                #                                                                       adjacency_matrix.shape[2],
                #                                                                       16)[:,:,:,[0, 1, 3, 8, 9, 13]]
                node_features = dpoint.x
                node_features = F.one_hot(node_features, 17).reshape(-1, 9, 17)
                node_features = node_features.reshape(-1, 9 * 17)

                self.pre_transformed_dataset.append((node_features, adjacency_matrix, dpoint.y))


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

        node_features = torch.zeros(batch_size, mx_nodes, n_node_features, device=self.device)
        adjacency_matrices = torch.zeros(batch_size, mx_nodes, mx_nodes, n_edge_features, device=self.device)
        mask = torch.ones(batch_size, mx_nodes, device=self.device)

        for i in range(batch_size):
            node_features[i,:nl[i],:] = nfs[i]
            adjacency_matrices[i,:nl[i],:nl[i],:] = ams[i]
            mask[i,:nl[i]] = 0.0

        return node_features, adjacency_matrices, torch.stack(lbs, 0), mask