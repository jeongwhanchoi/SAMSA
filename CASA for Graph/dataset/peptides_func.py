import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import random

class Peptide_func(nn.Module):
    def __init__(self, root):
        super().__init__()
        self.dataset = torch_geometric.datasets.LRGB(root, 'Peptides-func')
        self.d_len = len(self.dataset)
        self.indx = [i for i in range(self.d_len)]
        random.shuffle(self.indx)

    def get_graph(self):
        if len(self.indx) == 0:
            self.indx = [i for i in range(self.d_len)]
            random.shuffle(self.indx)
        node_feature = self.dataset[self.indx[:-1]]