import os
import math
import numpy as np
import cv2
import random

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
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box, LidarSegPointCloud

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def group_lidar_seg(label):
    # Replace the values in the label
    # print(label)
    grouped_label = group_lidar_seg.lookup_tensor[label.long()]
    mask_ignore = F.one_hot(grouped_label, num_classes=33)
    return grouped_label

# The dictionary with the replacement rules
replacement_dict = {
    0: 9,
    1: 14,
    2: 15,
    3: 17,
    4: 18,
    5: 21,
    6: 6,
    7: 12,
    8: 22,
    9: 23,
    10: 24,
    11: 25,
    12: 26,
    13: 27,
    14: 28,
    15: 30
 }

# Convert the dictionary to a list for mapping
mapping = [replacement_dict[i] for i in range(len(replacement_dict))]

# Create a lookup tensor
lookup_tensor = torch.tensor(mapping)

group_lidar_seg.lookup_tensor = lookup_tensor

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

class NuSceneLIDARSegmentationExperiment():
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

        train_dataset_settings = {
            "data_path": self.dataset_settings["data_path"],
            "version": version,
            "mode": "train"
        }

        val_dataset_settings = {
            "data_path": self.dataset_settings["data_path"],
            "version": version,
            "mode": "val"
        }

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

        val_dataloader_settings = {
            "dataset": self.val_dataset,
            "batch_size": self.dataset_settings["batch_size"],
            "shuffle": self.dataset_settings["shuffle"],
            "num_workers": self.dataset_settings["num_workers"],
            "drop_last": self.dataset_settings["drop_last"],
            "pin_memory": self.dataset_settings["pin_memory"],
        }

        self.train_dataloader = DataLoader(**train_dataloader_settings)
        self.val_dataloader = DataLoader(**val_dataloader_settings)

        model_settings = self.config['arch'].split('$')
        model_settings, model_type = model_settings[1], model_settings[0]
        self.model = wrapper[model_type](model_settings)
        self.model = self.model.to(self.device)
        # self.model = torch.compile(self.model, mode='max-autotune')
        self.model = torch.compile(self.model)

        self.weight = torch.zeros(16, device=self.device)

        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(self.val_dataloader):
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
                labels = F.one_hot(labels.reshape(-1).to(torch.long), num_classes=17)[:, :-1]
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
        
        self.metrics = parse_metrics_str(self.config['metrics']).to(self.device)
        self.best_performance = {}
        for i in ['miou', 'accuracy', 'macc']:
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
                inputs, labels, mask = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)
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
            eval_results = {}
            for key in range(len(self.metrics)):
                eval_results[self.metrics[key].settings["name"]] = None

            loop = tqdm(self.val_dataloader)
            whole_dataset_output = []
            whole_dataset_label = []
            whole_dataset_mask = []
            correct = torch.zeros(17, device=self.device)
            total = torch.zeros(17, device=self.device)
            intersect = torch.zeros(17, device=self.device)
            union = torch.zeros(17, device=self.device)
            with torch.no_grad():
                for i, data in enumerate(loop):
                    inputs, labels, mask = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    mask = mask.to(self.device)

                    x_coords = inputs[:, :, :3]
                    outputs = self.model(x=inputs, x_coords=x_coords, mask=mask)

                    outputs = torch.cat([outputs.reshape(-1, 16), (0.5 - mask.reshape(-1).unsqueeze(-1)) * 1e9], dim=1)
                    outputs = F.one_hot(torch.argmax(outputs, dim=1), num_classes=17)

                    labels = F.one_hot(labels.reshape(-1).to(torch.long), num_classes=17)

                    intersect = intersect + torch.sum(outputs * labels, dim=0)
                    union = union + torch.sum(torch.maximum(outputs, labels), dim=0)
                    correct = correct + torch.sum(outputs * labels, dim=0)
                    total = total + torch.sum(labels, dim=0)

                intersect = intersect[:16]
                union = union[:16]
                correct = correct[:16]
                total = total[:16]

                print("Intersect: ", intersect)
                print("Union: ", union)
                print("IoU: ", intersect / union)
                print("Correct: ", correct)
                print("Total: ", total)
                print("MACC: ", correct / total)


                miou = torch.sum(intersect / (union + 1e-9)) / torch.sum(torch.heaviside(union, torch.zeros(1, device=self.device)))
                accuracy = torch.sum(correct) / torch.sum(total)
                macc = torch.sum(correct / (total + 1e-9)) / torch.sum(torch.heaviside(total, torch.zeros(1, device=self.device)))
                eval_results = {
                    "miou": miou.item(),
                    "accuracy": accuracy.item(),
                    "macc": macc.item()
                }
                self.val_curve.append(eval_results)
            
            display_str = "Evaluation at Epoch " + str(self.current_epoch) + ":\n"
            for key in eval_results.keys():
                display_str += key + ": " + str(round_to_significant_digits(eval_results[key], 6)) + "\n"
                
            print(display_str)
            if epoch % self.save_frequency == 0:
                self.save_experiment()
            self.save_best(eval_results)
            self.current_epoch += 1
        return True
    
    def visualize(self, split='train'):
        for scene_idx in range(len(self.train_dataset.chosen_scene)):
            # if split == 'train':
            #     scene_index_token = self.train_dataset.chosen_scene[random.randint(0, len(self.train_dataset.chosen_scene) - 1)]
            # else:
            #     scene_index_token = self.val_dataset.chosen_scene[random.randint(0, len(self.val_dataset.chosen_scene) - 1)]
            scene_index_token = self.train_dataset.chosen_scene[scene_idx]
            data_path = self.dataset_settings["data_path"]
            scene = self.val_dataset.ns.nusc.scene[scene_index_token[0]]
            sample_in_scenes = []

            for sample in self.val_dataset.ns.nusc.sample:
                if sample["scene_token"] == scene["token"]:
                    sample_in_scenes.append(sample)

            for sample in sample_in_scenes:
                sample_data = sample['data']
                lidar_token = sample_data['LIDAR_TOP']
                file_name = self.val_dataset.ns.nusc.get('sample_data', lidar_token)['filename']
                fn = ""
                for lt in self.val_dataset.ns.nusc.lidarseg:
                    if lt["token"] == lidar_token:
                        fn = lt["filename"]
                        break
                point_cloud = LidarSegPointCloud(data_path + '/' + file_name, data_path + '/' + fn)
                points = torch.tensor(point_cloud.points)
                with torch.no_grad():
                    point_cloud = torch.tensor(points, dtype=torch.float32).unsqueeze(0).expand(self.dataset_settings["batch_size"], -1, -1).clone()
                    point_cloud = point_cloud.to(next(self.model.parameters()).device)

                    x_coords = point_cloud[:, :, :3]

                    predicted_labels = self.model(x=point_cloud, x_coords=x_coords, mask=torch.ones(self.dataset_settings["batch_size"], point_cloud.shape[1], device=point_cloud.device))
                    predicted_labels = torch.sign(predicted_labels_ - torch.max(predicted_labels_, dim=2, keepdim=True)[0])
                    predicted_labels = predicted_labels.sum(dim=0, keepdim=True)

                    predicted_labels = torch.argmax(predicted_labels, dim=2).squeeze(0)
                    predicted_labels = group_lidar_seg(predicted_labels.to('cpu')).detach().numpy()
                    
                bin_file_out =os.path.join(data_path +"/predictions/",lidar_token+"_lidarseg.bin")
                np.array(predicted_labels).astype(np.uint8).tofile(bin_file_out)
            
            curr_token= scene['first_sample_token']
            visited = set()
            images_path = []

            while True:
                if curr_token in visited:
                    break
                curr_sample = sample_in_scenes[0]
                for i in range(len(sample_in_scenes)):
                    sample = sample_in_scenes[i]
                    if sample["token"] ==  curr_token:
                        curr_sample = sample
                        break
                lidar_token = curr_sample["data"]['LIDAR_TOP']
                images_pred_path = images_pred_path = os.path.join(data_path, "image_predictions/" +lidar_token+"_lidarseg.png")
                images_path.append(images_pred_path)
                bin_file_out = os.path.join(data_path +"/predictions/",lidar_token+"_lidarseg.bin")
                self.val_dataset.ns.nusc.render_pointcloud_in_image(
                    curr_sample['token'],
                    pointsensor_channel='LIDAR_TOP',
                    camera_channel='CAM_BACK',
                    render_intensity=False,
                    show_lidarseg=True,
                    # filter_lidarseg_labels=[22, 23],
                    out_path = images_pred_path,
                    show_lidarseg_legend=True,
                    lidarseg_preds_bin_path=bin_file_out
                )
                visited.add(curr_token)
                curr_token = sample["next"]

            height, width, layers = 784, 1879, 3
            video_name = 'output' + str(scene_idx) + '.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video = cv2.VideoWriter(video_name, fourcc, 4, (width, height))
            repeat_count = 3

            # Iterate through the images and write them to the video
            for image in images_path:
                frame = cv2.imread(image)
                for _ in range(repeat_count):
                    video.write(frame)

            video.release()

        return True

class NuSceneLIDARSegmentationExperimentMultiGPU():
    def __init__(self,
                 exp_name,
                 real=True):
        device = 'cuda'
        self.device = 'cuda'
        self.exp_name = exp_name
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

        train_dataset_settings = {
            "data_path": self.dataset_settings["data_path"],
            "version": version,
            "mode": "train"
        }

        val_dataset_settings = {
            "data_path": self.dataset_settings["data_path"],
            "version": version,
            "mode": "val"
        }

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

        val_dataloader_settings = {
            "dataset": self.val_dataset,
            "batch_size": self.dataset_settings["batch_size"],
            "shuffle": self.dataset_settings["shuffle"],
            "num_workers": self.dataset_settings["num_workers"],
            "drop_last": self.dataset_settings["drop_last"],
            "pin_memory": self.dataset_settings["pin_memory"],
        }

        self.train_dataloader = DataLoader(**train_dataloader_settings)
        self.val_dataloader = DataLoader(**val_dataloader_settings)

        model_settings = self.config['arch'].split('$')
        model_settings, model_type = model_settings[1], model_settings[0]
        self.model = wrapper[model_type](model_settings).to(self.device)

        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(self.val_dataloader):
                inputs, labels, mask = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                mask = mask.to(device)
                x_coords = inputs[:, :, :3]
                outputs = self.model(x=inputs, x_coords=x_coords, mask=mask)
                break

        self.model = nn.parallel.DataParallel(self.model).to(self.device)

        self.lossfn = parse_lossfunction_str(self.config['loss_function'])
        self.lossfn = self.lossfn.to(self.device)
        self.optimizer, get_opt_fn = parse_optimizer_str(self.config['optimizer'])
        self.optimizer = get_opt_fn(self.model.parameters(), self.optimizer)
        self.scheduler, get_scheduler_fn = parse_scheduler_str(self.config['scheduler'])
        self.scheduler = get_scheduler_fn(self.optimizer, self.scheduler)
        
        self.metrics = parse_metrics_str(self.config['metrics']).to(self.device)
        self.best_performance = {}
        for i in ['miou', 'accuracy', 'macc']:
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
            'curve': self.train_curve,
        }
    
    def load_state(self, state):
        self.model.load_state_dict(state['model'], strict=False)
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.warmup_scheduler.load_state_dict(state['warmup_scheduler'])
        self.current_epoch = state['current_epoch']
        self.best_performance = state['best_performance']
        self.train_curve = state['curve']
        
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

        return laval_epoch
    
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
                inputs, labels, mask = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)
                self.optimizer.zero_grad()
                
                x_coords = inputs[:, :, :3]
                self.optimizer.zero_grad()
                outputs = self.model(x=inputs, x_coords=x_coords, mask=mask)
                loss = self.lossfn(inputs=outputs, targets=labels, mask=mask)
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
            eval_results = {}
            for key in range(len(self.metrics)):
                eval_results[self.metrics[key].settings["name"]] = None

            loop = tqdm(self.val_dataloader)
            whole_dataset_output = []
            whole_dataset_label = []
            whole_dataset_mask = []
            correct = torch.zeros(17, device=self.device)
            total = torch.zeros(17, device=self.device)
            intersect = torch.zeros(17, device=self.device)
            union = torch.zeros(17, device=self.device)
            with torch.no_grad():
                for i, data in enumerate(loop):
                    inputs, labels, mask = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    mask = mask.to(self.device)

                    x_coords = inputs[:, :, :3]
                    outputs = self.model(x=inputs, x_coords=x_coords, mask=mask)
                    
                    outputs = torch.cat([outputs.reshape(-1, 16), (0.5 - mask.reshape(-1).unsqueeze(-1)) * 1e9], dim=1)
                    outputs = F.one_hot(torch.argmax(outputs, dim=1), num_classes=17)

                    labels = F.one_hot(labels.reshape(-1).to(torch.long), num_classes=17)

                    intersect = intersect + torch.sum(outputs * labels, dim=0)
                    union = union + torch.sum(torch.maximum(outputs, labels), dim=0)
                    correct = correct + torch.sum(outputs * labels, dim=0)
                    total = total + torch.sum(labels, dim=0)

                intersect = intersect[:16]
                union = union[:16]
                correct = correct[:16]
                total = total[:16]

                print("Intersect: ", intersect)
                print("Union: ", union)


                miou = torch.sum(intersect / (union + 1e-9)) / torch.sum(torch.heaviside(union, torch.zeros(1, device=self.device)))
                accuracy = torch.sum(correct) / torch.sum(total)
                macc = torch.sum(correct / (total + 1e-9)) / torch.sum(torch.heaviside(total, torch.zeros(1, device=self.device)))
                eval_results = {
                    "miou": miou.item(),
                    "accuracy": accuracy.item(),
                    "macc": macc.item()
                }
                self.val_curve.append(eval_results)
            
            display_str = "Evaluation at Epoch " + str(self.current_epoch) + ":\n"
            for key in eval_results.keys():
                display_str += key + ": " + str(round_to_significant_digits(eval_results[key], 6)) + "\n"
                
            print(display_str)
            if epoch % self.save_frequency == 0:
                self.save_experiment()
            self.save_best(eval_results)
            self.current_epoch += 1
        return True
    
    def visualize(self):
        scene_index_token = self.val_dataset.chosen_scene[random.randint(0, len(self.val_dataset.chosen_scene) - 1)]
        data_path = self.dataset_settings["data_path"]
        scene = self.val_dataset.ns.nusc.scene[scene_index_token[0]]
        sample_in_scenes = []

        for sample in self.val_dataset.ns.nusc.sample:
            if sample["scene_token"] == scene["token"]:
                sample_in_scenes.append(sample)

        for sample in sample_in_scenes:
            sample_data = sample['data']
            lidar_token = sample_data['LIDAR_TOP']
            file_name = self.val_dataset.ns.nusc.get('sample_data', lidar_token)['filename']
            fn = ""
            for lt in self.val_dataset.ns.nusc.lidarseg:
                if lt["token"] == lidar_token:
                    fn = lt["filename"]
                    break
            point_cloud = LidarSegPointCloud(data_path + '/' + file_name, data_path + '/' + fn)
            points = torch.tensor(point_cloud.points)
            with torch.no_grad():
                point_cloud = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
                point_cloud = point_cloud.to(next(self.model.parameters()).device)
                predicted_labels = self.model(x=point_cloud, mask=torch.ones(1, point_cloud.shape[1], device=point_cloud.device))
                predicted_labels = torch.argmax(predicted_labels, dim=2).squeeze(0)
                predicted_labels = group_lidar_seg(predicted_labels.to('cpu')).detach().numpy()
                
            bin_file_out =os.path.join(data_path +"/predictions/",lidar_token+"_lidarseg.bin")
            np.array(predicted_labels).astype(np.uint8).tofile(bin_file_out)
      
        curr_token= scene['first_sample_token']
        visited = set()
        images_path = []

        while True:
            if curr_token in visited:
                break
            curr_sample = sample_in_scenes[0]
            for i in range(len(sample_in_scenes)):
                sample = sample_in_scenes[i]
                if sample["token"] ==  curr_token:
                    curr_sample = sample
                    break
            lidar_token = curr_sample["data"]['LIDAR_TOP']
            images_pred_path = images_pred_path = os.path.join(data_path, "image_predictions/" +lidar_token+"_lidarseg.png")
            images_path.append(images_pred_path)
            bin_file_out = os.path.join(data_path +"/predictions/",lidar_token+"_lidarseg.bin")
            self.val_dataset.ns.nusc.render_pointcloud_in_image(
                curr_sample['token'],
                pointsensor_channel='LIDAR_TOP',
                camera_channel='CAM_BACK',
                render_intensity=False,
                show_lidarseg=True,
                # filter_lidarseg_labels=[22, 23],
                out_path = images_pred_path,
                show_lidarseg_legend=True,
                lidarseg_preds_bin_path=bin_file_out
            )
            visited.add(curr_token)
            curr_token = sample["next"]

        height, width, layers = 784, 1879, 3
        video_name = 'output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))
        repeat_count = 3

        # Iterate through the images and write them to the video
        for image in images_path:
            frame = cv2.imread(image)
            for _ in range(repeat_count):
                video.write(frame)

        video.release()

        return True