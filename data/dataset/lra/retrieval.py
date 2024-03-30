import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pickle

class RetrievalDataset(Dataset):
    def __init__(self, split, path_folder='.\\data\\LRA\\'):
        path = path_folder + 'retrieval.' + split + '.pickle'
        self.data = pickle.load(open(path, 'rb'))
        self.num_label = 2

    def to_onehot(self, x):
        return F.one_hot(torch.tensor(x).to(torch.long), num_classes=128).to(torch.float32)

    def _get_item(self, index):
        dpoint = self.data[index]
        x1, x2, y = dpoint['input_ids_0'], dpoint['input_ids_1'], dpoint['label']
        x1, x2 = self.to_onehot(x1), self.to_onehot(x2)
        return x1, x2, y
    
    def __getitem__(self, index):
        return self._get_item(index)
    
    def __len__(self):
        return len(self.data)