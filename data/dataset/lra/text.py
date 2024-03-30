import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pickle

class TextClassificationDataset(Dataset):
    def __init__(self, split, path_folder='.\\data\\LRA\\'):
        path = path_folder + 'text.' + split + '.pickle'
        self.data = pickle.load(open(path, 'rb'))
        self.num_label = 2

    def to_onehot(self, x):
        return F.one_hot(torch.tensor(x).to(torch.long), num_classes=256).to(torch.float32)

    def _get_item(self, index):
        dpoint = self.data[index]
        x, y = dpoint['input_ids_0'], dpoint['label']
        x = self.to_onehot(x)
        return x, y
    
    def __getitem__(self, index):
        return self._get_item(index)
    
    def __len__(self):
        return len(self.data)