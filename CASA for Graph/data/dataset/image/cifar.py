import os
import json
import random
import pickle

import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

class CIFAR100Dataset(Dataset):
    def __init__(self, root, train=True, position=False, transform=None):
        cifar_dataset = torchvision.datasets.CIFAR100(root=root, train=train, transform=transform, download=True)
        self.transform = transform

        # Precompute the position matrix
        h, w = 32, 32
        self.position_matrix = np.zeros((h, w, 2), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                self.position_matrix[i, j, 0] = i / float(h)
                self.position_matrix[i, j, 1] = j / float(w)

        self.data = []
        for idx in range(len(cifar_dataset)):
            img, target = cifar_dataset[idx]

            # Concatenate RGB values and position features
            img = np.array(img)
            img = np.concatenate((img, self.position_matrix), axis=-1)

            # Flatten the 32x32x5 array to 1024x5
            img = img.reshape(-1, 5)

            # Normalize the color features (first 3 dimensions)
            img[:, :3] = (img[:, :3] - 0.5) * 2.0
            self.data.append((img, target))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx]
        return img, target
    
class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, position=False, transform=None):
        cifar_dataset = torchvision.datasets.CIFAR10(root=root, train=train, transform=transform, download=True)
        self.transform = transform

        # Precompute the position matrix
        h, w = 32, 32
        self.position_matrix = np.zeros((h, w, 2), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                self.position_matrix[i, j, 0] = i / float(h)
                self.position_matrix[i, j, 1] = j / float(w)

        self.data = []
        for idx in range(len(cifar_dataset)):
            img, target = cifar_dataset[idx]

            # Concatenate RGB values and position features
            img = np.array(img)
            if position == True:
                img = np.concatenate((img, self.position_matrix), axis=-1)

                # Flatten the 32x32x5 array to 1024x5
                img = img.reshape(-1, 5)

            else:
                # Flatten the 32x32x3 array to 1024x3
                img = img.reshape(-1, 3)

            # Normalize the color features (first 3 dimensions)
            img[:, :3] = (img[:, :3] - 0.5) * 2.0
            self.data.append((img, target))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx]
        return img, target