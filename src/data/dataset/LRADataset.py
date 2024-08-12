from ..name import dataset_parser_name_dict

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

class ListOPS(Dataset):
    def __init__(self, split, path_folder='.\\data\\LRA\\'):
        path = path_folder + 'listops.' + split + '.pickle'
        self.data = pickle.load(open(path, 'rb'))
        self.num_label = 10

    def to_onehot(self, x):
        return F.one_hot(torch.tensor(x).to(torch.long), num_classes=16).to(torch.float32)

    def _get_item(self, index):
        dpoint = self.data[index]
        x1, y = dpoint['input_ids_0'], dpoint['label']
        x1= self.to_onehot(x1)
        return x1, y
    
    def __getitem__(self, index):
        return self._get_item(index)
    
    def __len__(self):
        return len(self.data)

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

class PathFinderDataset(Dataset):
    def __init__(self, split, path_folder='.\\data\\LRA\\'):
        path = path_folder + 'pathfinder32-curv_contour_length_14.' + split + '.pickle'
        self.data = pickle.load(open(path, 'rb'))
        self.num_label = 2

    def _get_item(self, index):
        dpoint = self.data[index]
        x1, y = dpoint['input_ids_0'], dpoint['label']
        x1 = self.to_onehot(x1)
        return x1, y
    
    def __getitem__(self, index):
        return self._get_item(index)
    
    def __len__(self):
        return len(self.data)
    
    def to_onehot(self, x):
        return F.one_hot(torch.tensor(x).to(torch.long), num_classes=256).to(torch.float32)

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

dataset_parser_name_dict["IMDB"] = TextClassificationDataset
dataset_parser_name_dict["LISTOPS"] = ListOPS
dataset_parser_name_dict["CIFAR"] = CIFAR10Dataset
dataset_parser_name_dict["PATHFINDER"] = PathFinderDataset
dataset_parser_name_dict["RETRIEVAL"] = RetrievalDataset