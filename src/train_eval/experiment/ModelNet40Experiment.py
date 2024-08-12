import os
import math

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

class ModelNet40Experiment():
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

        train_dataset_settings = {
            "root": dataset_settings["root"],
            "num_category": dataset_settings["num_category"],
            "num_point":dataset_settings["num_point"],
            "use_uniform_sample": dataset_settings["use_uniform_sample"],
            "use_normals": dataset_settings["use_normals"],
            "split": "train"
        }

        val_dataset_settings = {
            "root": dataset_settings["root"],
            "num_category": dataset_settings["num_category"],
            "num_point":dataset_settings["num_point"],
            "use_uniform_sample": dataset_settings["use_uniform_sample"],
            "use_normals": dataset_settings["use_normals"],
            "split": "test"
        }

        train_dataset = dataset_fn(**train_dataset_settings)
        val_dataset = dataset_fn(**val_dataset_settings)

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

        self.train_dataloader = DataLoader(**train_dataloader_settings)
        self.val_dataloader = DataLoader(**val_dataloader_settings)
        
        model_settings = self.config['arch'].split('$')
        model_settings, model_type = model_settings[1], model_settings[0]
        self.model = wrapper[model_type](model_settings)
        self.model = self.model.to(self.device)
        self.model = torch.compile(self.model)

        self.weight = torch.zeros(40, device=self.device)

        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(self.train_dataloader):
                inputs, labels = data
                labels = labels.to(self.device)
                labels = F.one_hot(labels.reshape(-1).to(torch.long), num_classes=40)
                labels = torch.sum(labels, dim=0)
                self.weight = self.weight + labels
                break

        with torch.no_grad():
            self.model.eval()
            for i, data in enumerate(self.val_dataloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)                
                
                x_coords = inputs[:, :, :3] 

                inputs_dict = {
                    'x': inputs,
                    'x_coords': inputs[:, :, :3],
                    'mask': torch.ones(inputs.shape[0], inputs.shape[1], device=inputs.device),
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
        for i in ['accuracy', 'macc']:
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
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                x_coords = inputs[:, :, :3] 

                inputs_dict = {
                    'x': inputs,
                    'x_coords': inputs[:, :, :3],
                    'mask': torch.ones(inputs.shape[0], inputs.shape[1], device=inputs.device),
                    'labels': labels,
                }

                outputs = self.model(**inputs_dict)

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
            whole_dataset_output = []
            whole_dataset_label = []
            correct = torch.zeros(40, device=self.device)
            total = torch.zeros(40, device=self.device)
            with torch.no_grad():
                for i, data in enumerate(loop):
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    x_coords = inputs[:, :, :3] 

                    inputs_dict = {
                        'x': inputs,
                        'x_coords': inputs[:, :, :3],
                        'mask': torch.ones(inputs.shape[0], inputs.shape[1], device=inputs.device),
                        'labels': labels,
                    }

                    outputs = self.model(**inputs_dict)

                    outputs = outputs.reshape(-1, 40)
                    outputs = F.one_hot(torch.argmax(outputs, dim=1), num_classes=40)

                    labels = F.one_hot(labels.reshape(-1).to(torch.long), num_classes=40)

                    correct = correct + torch.sum(outputs * labels, dim=0)
                    total = total + torch.sum(labels, dim=0)

                correct = correct[:40]
                total = total[:40]

                print("Correct: ", correct)
                print("Total: ", total)
                print("MACC: ", correct / total)


                accuracy = torch.sum(correct) / torch.sum(total)
                macc = torch.sum(correct / (total + 1e-9)) / torch.sum(torch.heaviside(total, torch.zeros(1, device=self.device)))
                eval_results = {
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
    
    def evaluate(self, n_repeats=1):
        loop = tqdm(self.val_dataloader)
        whole_dataset_output = []
        whole_dataset_label = []
        correct = torch.zeros(40, device=self.device)
        total = torch.zeros(40, device=self.device)
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loop):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                x_coords = inputs[:, :, :3] 

                inputs_dict = {
                    'x': inputs,
                    'x_coords': inputs[:, :, :3],
                    'mask': torch.ones(inputs.shape[0], inputs.shape[1], device=inputs.device),
                    'labels': labels,
                }

                outputs = torch.zeros(inputs.shape[0], 40, device=inputs.device)
                for i in range(n_repeats):
                    outputs_ = self.model(**inputs_dict).reshape(-1, 40)
                    outputs_ = torch.sign(outputs_ - torch.max(outputs_, dim=-1, keepdim=True)[0])
                    outputs = outputs + outputs_

                outputs = F.one_hot(torch.argmax(outputs, dim=1), num_classes=40)

                labels = F.one_hot(labels.reshape(-1).to(torch.long), num_classes=40)

                correct = correct + torch.sum(outputs * labels, dim=0)
                total = total + torch.sum(labels, dim=0)

            correct = correct[:40]
            total = total[:40]


            accuracy = torch.sum(correct) / torch.sum(total)
            macc = torch.sum(correct / (total + 1e-9)) / torch.sum(torch.heaviside(total, torch.zeros(1, device=self.device)))
            eval_results = {
                "accuracy": accuracy.item(),
                "macc": macc.item()
            }

        return eval_results
