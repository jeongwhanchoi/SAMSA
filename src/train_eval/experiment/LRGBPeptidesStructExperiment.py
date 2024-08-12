import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAveragePrecision

from ...module import wrapper
from .utils import parse_settings_str
from ...data.dataset import parse_dataset_str
from ..loss_function import parse_lossfunction_str
from ...training_strategy.optimizer import parse_optimizer_str
from ...training_strategy.scheduler import parse_scheduler_str

from tqdm import tqdm

import pytorch_warmup as warmup

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

class LRGBPeptidesStructExperiment():
    def __init__(self,
                 exp_name,
                 device='cpu'):
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
        self.test_curve = []
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

        dataset_fn, dataset_settings = parse_dataset_str(self.config['dataloader'])

        self.metric = nn.L1Loss().to(self.device)

        train_dataset_settings = {
            "root": dataset_settings["root"],
            "pe": dataset_settings["pe"],
            "split": "train"
        }

        val_dataset_settings = {
            "root": dataset_settings["root"],
            "pe": dataset_settings["pe"],
            "split": "val"
        }

        test_dataset_settings = {
            "root": dataset_settings["root"],
            "pe": dataset_settings["pe"],
            "split": "test"
        }


        train_dataset = dataset_fn(**train_dataset_settings)
        val_dataset = dataset_fn(**val_dataset_settings)
        test_dataset = dataset_fn(**val_dataset_settings)

        train_dataloader_settings = {
            "dataset": train_dataset,
            "batch_size": dataset_settings["batch_size"],
            "shuffle": dataset_settings["shuffle"],
            "num_workers": dataset_settings["num_workers"],
            "drop_last": dataset_settings["drop_last"],
            "pin_memory": dataset_settings["pin_memory"],
        }

        val_dataloader_settings = {
            "dataset": val_dataset,
            "batch_size": dataset_settings["batch_size"],
            "shuffle": dataset_settings["shuffle"],
            "num_workers": dataset_settings["num_workers"],
            "drop_last": dataset_settings["drop_last"],
            "pin_memory": dataset_settings["pin_memory"],
        }

        test_dataloader_settings = {
            "dataset": test_dataset,
            "batch_size": dataset_settings["batch_size"],
            "shuffle": dataset_settings["shuffle"],
            "num_workers": dataset_settings["num_workers"],
            "drop_last": dataset_settings["drop_last"],
            "pin_memory": dataset_settings["pin_memory"],
        }

        self.train_dataloader = DataLoader(**train_dataloader_settings)
        self.val_dataloader = DataLoader(**val_dataloader_settings)
        self.test_dataloader = DataLoader(**test_dataloader_settings)
        
        model_settings = self.config['arch'].split('$')
        model_settings, model_type = model_settings[1], model_settings[0]
        self.model = wrapper[model_type](model_settings)
        self.model = self.model.to(self.device)
        self.model = torch.compile(self.model)

        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(self.val_dataloader):
                labels = data[-1].to(self.device)
                inputs_dict = {
                    'node_features': data[0].to(self.device), 
                    'edge_features': data[1].to(self.device), 
                    'mask_node': data[2].to(self.device), 
                    'mask_edge': data[3].to(self.device), 
                    'connection': data[4].to(self.device),
                    'labels': labels,
                }
                outputs = self.model(**inputs_dict)
                break

        self.lossfn = parse_lossfunction_str(self.config['loss_function'])
        self.lossfn = self.lossfn.to(self.device)
        self.optimizer, get_opt_fn = parse_optimizer_str(self.config['optimizer'])
        self.optimizer = get_opt_fn(self.model.parameters(), self.optimizer)
        self.scheduler, get_scheduler_fn = parse_scheduler_str(self.config['scheduler'])
        self.scheduler = get_scheduler_fn(self.optimizer, self.scheduler)
            
        self.best_performance = {}
        for i in ['MAE']:
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
            'test_curve': self.test_curve,
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
        self.test_curve = state['test_curve']
        
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
                labels = data[-1].to(self.device)
                inputs_dict = {
                    'node_features': data[0].to(self.device),
                    'edge_features': data[1].to(self.device),
                    'mask_node': data[2].to(self.device),
                    'mask_edge': data[3].to(self.device),
                    'connection': data[4].to(self.device),
                    'labels': labels,
                }
                outputs = self.model(**inputs_dict)

                self.optimizer.zero_grad()

                loss = self.lossfn(inputs=outputs, targets=labels)
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

            loop = tqdm(self.val_dataloader)
            val_MAE = 0.0
            val_MAE_bz = 0.0
            with torch.no_grad():
                for i, data in enumerate(loop):
                    labels = data[-1].to(self.device)
                    inputs_dict = {
                        'node_features': data[0].to(self.device),
                        'edge_features': data[1].to(self.device),
                        'mask_node': data[2].to(self.device),
                        'mask_edge': data[3].to(self.device),
                        'connection': data[4].to(self.device),
                        'labels': labels,
                    }
                    outputs = self.model(**inputs_dict)
                    MAE = self.metric(outputs,labels).detach().cpu().item()
                    val_MAE += MAE * outputs.shape[0]
                    val_MAE_bz += outputs.shape[0]

                val_MAE = val_MAE / val_MAE_bz

                eval_results = {
                    "MAE": val_MAE
                }
                self.val_curve.append(eval_results)
                eval_results["MAE"] *= -1

            loop = tqdm(self.test_dataloader)
            test_MAE = 0.0
            test_MAE_bz = 0.0
            with torch.no_grad():
                for i, data in enumerate(loop):
                    labels = data[-1].to(self.device)
                    inputs_dict = {
                        'node_features': data[0].to(self.device),
                        'edge_features': data[1].to(self.device),
                        'mask_node': data[2].to(self.device),
                        'mask_edge': data[3].to(self.device),
                        'connection': data[4].to(self.device),
                        'labels': labels,
                    }
                    outputs = self.model(**inputs_dict)
                    MAE = self.metric(outputs, labels).detach().cpu().item()
                    test_MAE += MAE * outputs.shape[0]
                    test_MAE_bz += outputs.shape[0]

                test_MAE = test_MAE / test_MAE_bz

                test_results = {
                    "MAE": test_MAE
                }
                self.test_curve.append(test_results)
                test_results["MAE"] *= -1
            
            display_str = "Evaluation at Epoch " + str(self.current_epoch) + ":\n"
            for key in eval_results.keys():
                display_str += key + ": " + str(round_to_significant_digits(eval_results[key], 6)) + "\n"

            display_str += "Test at Epoch " + str(self.current_epoch) + ":\n"
            for key in test_results.keys():
                display_str += key + ": " + str(round_to_significant_digits(test_results[key], 6)) + "\n"
                
            print(display_str)
            if epoch % self.save_frequency == 0:
                self.save_experiment()
            self.save_best(eval_results)
            self.current_epoch += 1
        return True