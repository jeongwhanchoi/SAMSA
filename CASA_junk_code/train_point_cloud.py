import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
import pytorch_warmup as warmup
from pytorch_lamb import Lamb, log_lamb_rs
import numpy as np
import random
import time
from copy import deepcopy
from tqdm import tqdm
from multimethod import multimethod
import argparse
import logging
from model import parse_point_architecture_mtl
from data.dataset import ModelNetDataLoader, PartNormalDataset
import warnings
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

class train_state():
    def __init__(self):
        root = './datasets'
        modelnet_data_path = os.path.join(root, "modelnet40_normal_resampled")
        self.modelnet_train_dataset = ModelNetDataLoader(modelnet_data_path, 
                                num_category=40,
                                split='train', 
                                process_data=True,
                                transforms=False,
                                use_uniform_sample=True,
                                use_normals=True,
                                num_point=1024)

        self.modelnet_test_dataset = ModelNetDataLoader(modelnet_data_path, 
                                num_category=40,
                                split='test', 
                                process_data=True,
                                transforms=False,
                                use_uniform_sample=True,
                                use_normals=True,
                                num_point=1024)

        self.modelnet_trainDataLoader = torch.utils.data.DataLoader(self.modelnet_train_dataset, batch_size=64, shuffle=True, num_workers=8, drop_last=True)
        self.modelnet_testDataLoader = torch.utils.data.DataLoader(self.modelnet_test_dataset, batch_size=64, shuffle=False, num_workers=8)

        self.shapenet_train_dataset = PartNormalDataset(root = './datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                                            npoints=2500, 
                                            split='train', 
                                            class_choice=None, 
                                            normal_channel=True)
        self.shapenet_trainDataLoader = torch.utils.data.DataLoader(self.shapenet_train_dataset, 
                                                        batch_size=32, 
                                                        shuffle=True, 
                                                        num_workers=8, 
                                                        drop_last=True)

        self.shapenet_test_dataset = PartNormalDataset(root = './datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                                            npoints=2500, 
                                            split='test', 
                                            class_choice=None, 
                                            normal_channel=True)
        self.shapenet_testDataLoader = torch.utils.data.DataLoader(self.shapenet_test_dataset, 
                                                        batch_size=32, 
                                                        shuffle=False, 
                                                        num_workers=8)
        self.m40_data = self.get_batch('modelnet')
        self.shapenet_data = self.get_batch('shapenet')
        
        model_str = "PCLL,128"
        for i in range(8):
            model_str = model_str + "@RSTL,128,8,256,256,0.1,False,512,0.1,composite,False,LayerNorm,gelu"
        self.model = parse_point_architecture_mtl(model_str, is_normal=True, is_rotation=False)
        self.model.append_task(nn.LazyLinear(40))
        self.model.append_task(nn.LazyLinear(50))
        self.model = self.model.cuda()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                weight_decay=0.1
            )
        self.warmup_scheduler  = warmup.UntunedLinearWarmup(self.optimizer)
        self.lr_scheduler      = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.9)
        self.number_of_updated_step = 0
        self.max_m40 = -10
        self.max_shapenet_ins = -10
        self.max_shapenet_mean = -10
        self.history = []

    def get_batch(self, dataset):
        if dataset == 'modelnet':
            loop = self.modelnet_trainDataLoader
            data = []
            for points, target in loop:
                data.append((points, target))
            return data
        elif dataset == 'shapenet':
            loop = self.shapenet_trainDataLoader
            data = []
            for points, _, target in loop:
                data.append((points, target))
            return data
        raise ValueError
    
    def get_dpoint(self, dataset):
        if dataset == 'modelnet':
            if len(self.m40_data) == 0:
                self.m40_data = self.get_batch(dataset)
            return self.m40_data.pop()
        if dataset == 'shapenet':
            if len(self.shapenet_data) == 0:
                self.shapenet_data = self.get_batch(dataset)
            return self.shapenet_data.pop()
        raise ValueError

    def optimize(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.number_of_updated_step % 100 != 0:
            self.warmup_scheduler.dampening()
        else:
            with self.warmup_scheduler.dampening():
                self.lr_scheduler.step()

    def step_modelnet(self):
        self.number_of_updated_step += 1
        points, target = self.get_dpoint('modelnet')
        points, target = points.cuda(), target.cuda()
        points = points.to(torch.float32)
        target_oh = F.one_hot(target.long(), num_classes=40).to(points.dtype)
        self.model.set_task(0)
        pred = self.model(points)
        pred = torch.max(pred, dim=1)[0]
        loss = self.criterion(pred, target_oh)

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        accuracy = correct.item() / float(points.size()[0])

        (loss / 2.0).backward()
        del points
        del target
        return accuracy
    
    def step_shapenet(self):
        self.number_of_updated_step += 1
        points, target = self.get_dpoint('shapenet')
        points, target = points.cuda(), target.cuda()
        self.model.set_task(1)
        pred = self.model(points)
        loss = self.criterion(pred.reshape(-1, 50), target.reshape(-1).long())
        (loss / 2.0).backward()
        del points
        del target
        return loss.item()

    def validate(self):
        self.model = self.model.train()
        self.model.set_task(0)

        with torch.no_grad():
            test_mean_correct = []
            test_class_acc = np.zeros((40, 3))
            loop = self.modelnet_testDataLoader
            for points, target in loop:
                points, target = points.cuda(), target.cuda()
                points = points.to(torch.float32)
                pred = self.model(points)
                pred = torch.max(pred, dim=1)[0]
                pred_choice = pred.data.max(1)[1]
                for cat in np.unique(target.cpu()):
                    classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                    test_class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
                    test_class_acc[cat, 1] += 1
                correct = pred_choice.eq(target.long().data).cpu().sum()
                test_mean_correct.append(correct.item() / float(points.size()[0]))
            test_class_acc[:, 2] =  test_class_acc[:, 0] / test_class_acc[:, 1]
            test_class_acc = np.mean(test_class_acc[:, 2])
            modelnet_test_instance_acc = np.mean(test_mean_correct)
        
        if modelnet_test_instance_acc > self.max_m40:
            self.save("modelnet")
            self.max_m40 = modelnet_test_instance_acc

        seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        
        self.model.set_task(1)

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(50)]
            total_correct_class = [0 for _ in range(50)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            for batch_id, (points, label, target) in enumerate(self.shapenet_testDataLoader):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                seg_pred = self.model(points)

                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(50):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)

            if test_metrics['instance_avg_iou'] > self.max_shapenet_ins:
                self.save("ins_shapenet")
                self.max_shapenet_ins = test_metrics['instance_avg_iou']

            if test_metrics['class_avg_iou'] > self.max_shapenet_mean:
                self.save("class_avg_shapenet")
                self.max_shapenet_mean = test_metrics['class_avg_iou']

            print("Modelnet:", modelnet_test_instance_acc, "\nShapenet ins:", test_metrics['instance_avg_iou'], "\nShapenet cls", test_metrics['class_avg_iou'])
            print("Bestm40:", self.max_m40, "Bestshape ins:", self.max_shapenet_ins, "Bestshape cls", self.max_shapenet_mean)
            self.history.append((modelnet_test_instance_acc, test_metrics['instance_avg_iou'], test_metrics['class_avg_iou']))

    def save(self, dataset):
        torch.save(self.model.state_dict(), "./mtl_checkpoints/" + dataset + ".pth")

    def train(self, n_steps):
        loop = tqdm(range(n_steps))
        self.model = self.model.train()
        for i in loop:
            a1 = self.step_modelnet()
            a2 = self.step_shapenet()

            loop.set_description_str("m40: " + str(a1)[:7] + " shapenet: " + str(a2)[:7])

            self.optimize()

            if self.number_of_updated_step % 100 == 0:
                self.model = self.model.eval()
                self.validate()
                self.model = self.model.train()
        return None

train_s = train_state()
train_s.train(45900)