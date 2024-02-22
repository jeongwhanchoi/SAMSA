import os
import json
import random
import pickle

import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from tqdm import tqdm
from data.augmentation import *
from data.utils import pc_normalize, centralize, farthest_point_sample

class ModelNetDataLoader(Dataset):
    def __init__(self, root, 
                 num_point=1024, 
                 transforms=True, 
                 use_uniform_sample=False, 
                 use_normals=True,
                 num_category=40, 
                 split='train', 
                 process_data=False,
                 cache_size=15000):
        self.root = root
        self.n_sample = num_point
        self.process_data = process_data
        self.uniform = use_uniform_sample
        self.use_normals = use_normals
        self.num_category = num_category
        self.transforms = transforms

        if self.num_category == 10:
            self.catfile = os.path.join(
                self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(
                self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(
                os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(
                os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(
                os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(
                os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (
                self.num_category, split, self.n_sample))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (
                self.num_category, split, self.n_sample))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' %
                      self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath), position=0, leave=True):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(
                        fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(
                            point_set, self.n_sample)
                    else:
                        point_set = ShufflePoints()(point_set)
                        point_set = point_set[0:self.n_sample, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

        self.cache_size = cache_size # number of datapoints to cache memory
        self.cache = {} # from index to (point_set, cls) tuple
        
    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        retrieve_cache_flag = False
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
            
            if not self.use_normals:
                point_set = point_set[:, :3]

            if self.transforms is True:
                point_set[:, :3] = rotate_point_cloud(point_set[:, :3])
        else:
            if index in self.cache:
                point_set, cls = self.cache[index]
                retrieve_cache_flag = True
            else:
                fn = self.datapath[index]
                cls = self.classes[self.datapath[index][0]]
                label = np.array([cls]).astype(np.int32)
                point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
                
                if len(self.cache) < self.cache_size:
                    self.cache[index] = (point_set, label[0])
                    
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.n_sample)
            else:
                point_set = ShufflePoints(point_set)
                point_set = point_set[0:self.n_sample, :]
            if not self.use_normals:
                point_set = point_set[:, :3]

            if self.transforms is True:
                # point_set = random_point_dropout(point_set)
                point_set[:, :3] = rotate_point_cloud(point_set[:, :3])
        
        if retrieve_cache_flag:
            return point_set, cls
        else:  
            return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)