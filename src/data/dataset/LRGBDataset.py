import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm

from ..name import dataset_parser_name_dict

import os
from compress_pickle import dump, load
from torch.utils.data import Dataset

class PeptideStructDataset(Dataset):
    def __init__(self, root, pe, split):
        super().__init__()
        if pe is True:
            file = os.path.join(root, "lrgb_peptides_struct_" + split + "_rwpe_.lzma")
        elif pe is False:
            file = os.path.join(root, "lrgb_peptides_struct_" + split + "_None_.lzma")
        else:
            raise ValueError

        self.pre_transformed_dataset = load(file, compression="lzma")

    def __getitem__(self, index):
        temp = self.pre_transformed_dataset[index]
        return temp[0], temp[1], temp[2].squeeze(-1), temp[3].squeeze(-1), temp[4], temp[5].squeeze(0)

    def __len__(self):
        return len(self.pre_transformed_dataset)

class PeptideFuncDataset(Dataset):
    def __init__(self, root, pe, split):
        super().__init__()
        if pe is True:
            file = os.path.join(root, "lrgb_peptides_func_" + split + "_rwpe_.lzma")
        elif pe is False:
            file = os.path.join(root, "lrgb_peptides_func_" + split + "_None_.lzma")
        else:
            raise ValueError

        self.pre_transformed_dataset = load(file, compression="lzma")

    def __getitem__(self, index):
        temp = self.pre_transformed_dataset[index]
        return temp[0], temp[1], temp[2].squeeze(-1), temp[3].squeeze(-1), temp[4], temp[5].squeeze(0)

    def __len__(self):
        return len(self.pre_transformed_dataset)

dataset_parser_name_dict["LRGBSTRUCT"] = PeptideStructDataset
dataset_parser_name_dict["LRGBFUNC"] = PeptideFuncDataset