import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pickle

class CIFAR10Dataset(Dataset):
    def __init__(self, split, path_folder='./datasets/LRA/'):
        path = path_folder + 'image.' + split + '.pickle'
        self.data = pickle.load(open(path, 'rb'))
        self.num_label = 10

    def _get_item(self, index):
        dpoint = self.data[index]
        x1, y = dpoint['input_ids_0'], dpoint['label']
        x1= self.to_onehot(x1)
        return x1, y
    
    def __getitem__(self, index):
        return self._get_item(index)
    
    def __len__(self):
        return len(self.data)
    
    def to_onehot(self, x):
        return F.one_hot(torch.tensor(x).to(torch.long), num_classes=256).to(torch.float32)