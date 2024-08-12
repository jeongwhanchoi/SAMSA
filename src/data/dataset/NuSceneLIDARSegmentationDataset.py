from ..name import dataset_parser_name_dict

from ..data_processing.NuSceneData import NuSceneData
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F

def group_lidar_seg(label):
    # Replace the values in the label
    grouped_label = group_lidar_seg.lookup_tensor[label.long()]
    mask_ignore = F.one_hot(grouped_label, num_classes=17)
    return grouped_label, 1 - mask_ignore[..., -1]

# The dictionary with the replacement rules
replacement_dict = {
    0: 16,
    1: 16,
    2: 6,
    3: 6,
    4: 6,
    5: 16,
    6: 6,
    7: 16,
    8: 16,
    9: 0,
    10: 0,
    11: 16,
    12: 7,
    13: 16,
    14: 1,
    15: 2,
    16: 2,
    17: 3,
    18: 4,
    19: 16,
    20: 16,
    21: 5,
    22: 8,
    23: 9,
    24: 10,
    25: 11,
    26: 12,
    27: 13,
    28: 14,
    29: 16,
    30: 15,
    31: 16
}

# Convert the dictionary to a list for mapping
mapping = [replacement_dict[i] for i in range(len(replacement_dict))]

# Create a lookup tensor
lookup_tensor = torch.tensor(mapping)

group_lidar_seg.lookup_tensor = lookup_tensor

class NuSceneLIDARSegmentationDataset(Dataset):
    def __init__(self, data_path="data/nuscenes", version="v1.0-mini", mode="train"):
        """
            NuScene LIDAR Point Cloud Segmentation TorchDataset Implementation.
            hyperparameters:
                data_path (str): the path to nuscene dataset; the nuscene dataset should consist of the following folders:
                                 maps, samples, sweeps, ...

                                 
                version (str): the set of data we are using; there are three available sets of data:
                                + trainval split: to choose this split, use version="v1.0-trainval"
                                + test split: to choose this split, use version="v1.0-test"
                                + mini split: to choose this split, use version="v1.0-mini"

                mode (str): train/validation split

            details:
                    each scece consists of roughly 458.8 samples
                    trainval split consists of 850 scenes
                    test split consists of 150 scenes
                    each sample consists of roughly 34k points
        """
        super().__init__()
        self.ns = NuSceneData(data_path, version)
        self.pcs = []
        self.lbs = []
        self.npoints = []

        current_random_state = random.getstate()
        current_random_state = current_random_state[1][0]

        random.seed(5622023)
        self.chosen_scene = list(self.ns.scene_token_list.keys())
        self.chosen_scene = [(i, self.chosen_scene[i]) for i in range(len(self.chosen_scene))]
        random.shuffle(self.chosen_scene)
        n_scenes = len(self.chosen_scene)
        n_train_scenes = int(0.85 * n_scenes)
        

        if mode == 'train':
            self.chosen_scene = self.chosen_scene[:n_train_scenes]
        elif mode == 'val':
            self.chosen_scene = self.chosen_scene[n_train_scenes:]

        chosen_scene_dict = dict()
        for i in self.chosen_scene:
            chosen_scene_dict[i[1]] = 1
        random.seed(current_random_state)

        sample_token_list = list(self.ns.sample_token.keys())
        sample_token_scene = [chosen_scene_dict.get(self.ns.sample_token[i]["scene_token"], 0) for i in sample_token_list]

        self.mxshape = 0

        for i in tqdm(range(len(sample_token_list))):
            if sample_token_scene[i] == 1:
                pc, lb = self.ns.get_lidar_point_cloud(sample_token=sample_token_list[i])
                lb = torch.tensor(lb)
                lb, mask = group_lidar_seg(lb)
                idx = torch.argsort(mask, dim=-1, stable=True, descending=True)
                self.pcs.append(torch.tensor(pc)[idx, :])
                self.lbs.append(torch.tensor(lb)[idx])
                self.npoints.append(pc.shape[0])
                self.mxshape = max(self.mxshape, pc.shape[0])

        self.d = self.pcs[0].shape[1]

    def __len__(self):
        return len(self.pcs)

    def __getitem__(self, i):
        pcs = torch.zeros(self.mxshape, self.pcs[i].shape[1])
        lbs = torch.ones(size=tuple([self.mxshape])) * 16
        mask = torch.zeros(self.mxshape)
        pcs[:self.pcs[i].shape[0]] = self.pcs[i]
        lbs[:self.lbs[i].shape[0]] = self.lbs[i]
        mask[:self.npoints[i]] = 1.0
        return pcs, lbs, mask
    
dataset_parser_name_dict["NUSCENELIDARSEGMENTATION"] = NuSceneLIDARSegmentationDataset
