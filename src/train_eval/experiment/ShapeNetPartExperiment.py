import os
import math
import numpy as np
import cv2
import random

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...module import wrapper
from .utils import parse_settings_str
from ...data.dataset import parse_dataset_str
from ..loss_function import parse_lossfunction_str
from ...training_strategy.optimizer import parse_optimizer_str
from ...training_strategy.scheduler import parse_scheduler_str

from ..loss_function import mIOU

from tqdm import tqdm

import pytorch_warmup as warmup

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def round_to_significant_digits(num, digits):
    if num == 0:
        return 0
    else:
        # Determine the scale factor
        scale = math.floor(math.log10(abs(num)))
        # Shift the number to have the desired significant digits before the decimal
        shifted = num / (10**scale)
        # Round the shifted number
        rounded = round(shifted, digits - 1)
        # Shift back
        return rounded * (10**scale)

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(50)[y.cpu().data.numpy(),]
    return new_y.to(y.device)


class ShapeNetPartSegmentationExperiment():
    def __init__(self,
                 exp_name,
                 device='cpu',
                 real=True):
        self.exp_name = exp_name
        self.device = device
        self.path = os.path.join(os.getcwd(), "experiment")
        self.path = os.path.join(self.path, exp_name)

        self.setting_path = os.path.join(self.path, "settings.txt")
        self.checkpoint_path = os.path.join(self.path, "checkpoints")
        self.current_epoch = 0

        self.config = {}
        current_section = None
        self.train_curve = []
        self.val_curve = []
        with open(self.setting_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('begin'):
                    current_section = line.split()[1:]
                    if len(current_section) == 1:
                        self.config[current_section[0]] = ""
                        current_section = current_section[0]
                    else:
                        self.config[current_section[0]] = current_section[1] + '$'
                        current_section = current_section[0]
                elif line.startswith('end'):
                    current_section = None
                elif current_section:
                    self.config[current_section] += line + '\n'

        for key in self.config:
            if self.config[key][-1] == '\n':
                self.config[key] = self.config[key][:-1]

        dataset_fn, self.dataset_settings = parse_dataset_str(self.config['dataloader'])
        if real is True:
            version = "v1.0-trainval"
        else:
            version = "v1.0-mini"

        train_dataset_settings = deepcopy(self.dataset_settings)

        val_dataset_settings = deepcopy(self.dataset_settings)

        train_dataset_settings["split"] = 'trainval'
        val_dataset_settings["split"] = 'test'

        self.train_dataset = dataset_fn(**train_dataset_settings)
        self.val_dataset = dataset_fn(**val_dataset_settings)

        train_dataloader_settings = {
            "dataset": self.train_dataset,
            "batch_size": self.dataset_settings["batch_size"],
            "shuffle": self.dataset_settings["shuffle"],
            "num_workers": self.dataset_settings["num_workers"],
            "drop_last": self.dataset_settings["drop_last"],
            "pin_memory": self.dataset_settings["pin_memory"],
        }

        test_dataloader_settings = {
            "dataset": self.val_dataset,
            "batch_size": self.dataset_settings["batch_size"],
            "shuffle": self.dataset_settings["shuffle"],
            "num_workers": self.dataset_settings["num_workers"],
            "drop_last": self.dataset_settings["drop_last"],
            "pin_memory": self.dataset_settings["pin_memory"],
        }

        self.train_dataloader = DataLoader(**train_dataloader_settings)
        self.test_dataloader = DataLoader(**test_dataloader_settings)

        model_settings = self.config['arch'].split('$')
        model_settings, model_type = model_settings[1], model_settings[0]
        self.model = wrapper[model_type](model_settings)
        self.model = self.model.to(self.device)
        # self.model = torch.compile(self.model, mode='max-autotune')
        self.model = torch.compile(self.model)

        self.weight = torch.zeros(50, device=self.device)

        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(self.test_dataloader):
                inputs, labels, mask = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                mask = mask.to(device)
                x_coords = inputs[:, :, :3]
                outputs = self.model(x=inputs, x_coords=x_coords, mask=mask)
                break
            
            for i, data in enumerate(self.train_dataloader):
                inputs, labels, mask = data
                labels = labels.to(self.device)
                labels = F.one_hot(labels.reshape(-1).to(torch.long), num_classes=51)[:, :-1]
                labels = torch.sum(labels, dim=0)
                self.weight = self.weight + labels
                break

            self.weight = torch.sum(self.weight) / self.weight

        self.lossfn = parse_lossfunction_str(self.config['loss_function'])
        self.lossfn = torch.compile(self.lossfn.to(self.device))
        self.optimizer, get_opt_fn = parse_optimizer_str(self.config['optimizer'])
        self.optimizer = get_opt_fn(self.model.parameters(), self.optimizer)
        self.scheduler, get_scheduler_fn = parse_scheduler_str(self.config['scheduler'])
        self.scheduler = get_scheduler_fn(self.optimizer, self.scheduler)

        self.best_performance = {}
        for i in ['accuracy', 'class_avg_accuracy', 'class_avg_iou', 'instance_avg_iou']:
            self.best_performance[i] = -1e9

        self.misc_settings = parse_settings_str(self.config['miscellaneous'])
        self.number_of_epoch = self.misc_settings['number_of_epoch']
        self.save_frequency = self.misc_settings['save_frequency']
        self.warmup_scheduler = warmup.LinearWarmup(self.optimizer, warmup_period=self.misc_settings['number_of_warmup_step'])
        self.max_gradient_norm = self.misc_settings['max_gradient_norm']

    def get_state(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'warmup_scheduler': self.warmup_scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'best_performance': self.best_performance,
            'train_curve': self.train_curve,
            'val_curve': self.val_curve,
        }
    
    def load_state(self, state):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.warmup_scheduler.load_state_dict(state['warmup_scheduler'])
        self.current_epoch = state['current_epoch']
        self.best_performance = state['best_performance']
        self.train_curve = state['train_curve']
        self.val_curve = state['val_curve']
        
    def save_experiment(self):
        torch.save(self.get_state(), 
                   os.path.join(self.checkpoint_path, "state_" + "epoch_" + str(self.current_epoch) + ".pth"))
        return True
    
    def save_best(self, perf):
        for key in perf.keys():
            if self.best_performance[key] < perf[key]:
                self.best_performance[key] = perf[key]
                torch.save(self.get_state(), 
                        os.path.join(self.checkpoint_path, "state_best_" + key + "_epoch_" + str(self.current_epoch) + ".pth"))
        torch.save(self.get_state(), 
                   os.path.join(self.checkpoint_path, "state_latest.pth"))
        return True
    
    def load_experiment(self):
        # Load the state dicts of the laval epoch
        state = torch.load(os.path.join(self.checkpoint_path, f'state_latest.pth'), map_location=self.device)
        self.load_state(state)

        return None
    
    def load_experiment_from_file(self, file_name):
        # Load the state dicts of the laval epoch
        state = torch.load(os.path.join(self.checkpoint_path, file_name), map_location=self.device)
        self.load_state(state)
        return None

    def run_experiment(self):
        for epoch in range(self.current_epoch, self.number_of_epoch):
            self.model.train()
            loop = tqdm(self.train_dataloader)
            
            eval_results = {}
            eval_results[self.lossfn.settings["name"]] = []
            
            for i, data in enumerate(loop):
                inputs, _, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                mask = torch.ones(inputs.shape[0], inputs.shape[1], device=inputs.device)
                x_coords = inputs[:, :, :3]
                self.optimizer.zero_grad()
                outputs = self.model(x=inputs, x_coords=x_coords, mask=mask)
                loss = self.lossfn(inputs=outputs, targets=labels, mask=mask, weight=self.weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.misc_settings['max_gradient_norm'])
                self.optimizer.step()
                with torch.no_grad():
                    metrics_measure = {}
                    metrics_measure[self.lossfn.settings["name"]] = loss.item()
                    eval_results[self.lossfn.settings["name"]].append(loss.item())
                    self.train_curve.append(metrics_measure)
                    display_str = "Epoch " + str(self.current_epoch) + ";"
                    for key in metrics_measure:
                        display_str += key + ": " + str(round_to_significant_digits(metrics_measure[key], 6)) + ", "
                loop.set_description_str(display_str)
                if i < len(self.train_dataloader) - 1:
                    with self.warmup_scheduler.dampening():
                        pass
            
            display_str = "Training at Epoch " + str(self.current_epoch) + ":\n"
            for key in eval_results.keys():
                lst = eval_results[key]
                mean = sum(lst) / len(lst)
                eval_results[key] = mean
                display_str += key + ": " + str(round_to_significant_digits(mean, 6)) + "\n"
                
            print(display_str)

            with self.warmup_scheduler.dampening():
                self.scheduler.step()
            
            self.model.eval()
            loop = tqdm(enumerate(self.test_dataloader))
            
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

            with torch.no_grad():
                for batch_id, (points, labels, target) in loop:
                    cur_batch_size, NUM_POINT, _ = points.size()
                    points, labels, target = points.float(), labels.long(), target.long()
                    points = points.to(self.device)
                    labels = labels.to(self.device)
                    target = target.to(self.device)
                    x_coords = points[:, :, :3]
                    mask = torch.ones(points.shape[0], points.shape[1], device=points.device)
                    seg_pred = self.model(x=points, x_coords=x_coords, mask=mask)

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
                test_metrics = {}
                test_metrics['accuracy'] = total_correct / float(total_seen)
                test_metrics['class_avg_accuracy'] = np.mean(
                    np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
                for cat in sorted(shape_ious.keys()):
                    print('Eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
                test_metrics['class_avg_iou'] = mean_shape_ious
                test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)

                print(test_metrics)
                self.val_curve.append(test_metrics)
            
            display_str = "Evaluation at Epoch " + str(self.current_epoch) + ":\n"
            if epoch % self.save_frequency == 0:
                self.save_experiment()
            self.save_best(test_metrics)
            self.current_epoch += 1
        return True