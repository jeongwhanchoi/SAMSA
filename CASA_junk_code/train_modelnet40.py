### IMPORT LIBRARIES

## AUTOMATIC DIFFERENTIATION
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
from model import parse_point_architecture
from data.dataset import ModelNetDataLoader
import warnings
warnings.filterwarnings('ignore')

# torch.set_num_threads(4)
torch.autograd.set_detect_anomaly(True)

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def create_log_file(save_dir, log_file_name):
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_file_name)
    return open(log_file_path, 'a')


def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for Training Progress")

    # Add arguments
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--root", type=str, default="./datasets", help="Dataset directory folder.")
    parser.add_argument("--uniform", required=True, help="Whether to use farthest point sampling or not. (True/False).")
    parser.add_argument("--model_str", type=str, help="Architecture, example: PCLL,96@RESC,96,96,8,192,0.1,1,True@RESC,96,96,8,192,0.1,1,True@RESC,96,96,8,192,0.1,1,True@CLFH,40")
    parser.add_argument("--model_name", type=str, help="Name to save model.")
    parser.add_argument("--is_rotation", type=str, help=".")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, required=False, help="Weight decay value of optimize function")
    parser.add_argument("--seed", type=int, default=1, help="Training seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path to training checkpoints")
    parser.add_argument("--exp_name", type=str, default="wo_fps_depth_3", help="Experiment name")
    parser.add_argument("--num_points", type=int, default=1024, help="Number of sampled points.")
    parser.add_argument("--use_normal", type=str, help=".")
    parser.add_argument("--accumulate", type=int, help=".")

    
    return parser.parse_args()

def augment(pc, is_rotation):
    with torch.no_grad():
        point_cloud = pc.clone()
        batch_size, n_points, _ = point_cloud.shape
        device = point_cloud.device  # Get the device from the input point cloud
        
        point_cloud[:,:,:3] = point_cloud[:,:,:3] - torch.mean(point_cloud[:,:,:3], dim=1, keepdim=True)

        # """Randomly jittering point cloud"""
        # point_cloud[:,:,:3] = point_cloud[:,:,:3] * ((torch.rand(point_cloud.shape[0], 1, 3, device=device) - 0.5) * 2 * 0.001)

        # """Randomly drop normal vectors"""
        # normal_drop = torch.bernoulli(torch.ones(point_cloud.shape[0], point_cloud.shape[1], 3, device=device) * 0.9)
        # point_cloud[:,:,3:] = point_cloud[:,:,3:] * normal_drop
        
        centroid = torch.mean(point_cloud[:,:,:3], dim=1, keepdim=True)
        point_cloud[:,:,:3] = point_cloud[:,:,:3] - centroid
        # max_distance = torch.max(torch.sqrt(torch.sum((point_cloud[:,:,:3] - centroid) ** 2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0] * ((torch.rand(point_cloud.shape[0], 1, 3, device=device) - 0.5) * 0.4 + 1)
        # max_distance = torch.max(torch.sqrt(torch.sum((point_cloud[:,:,:3] - centroid) ** 2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0]
        # max_distance = (torch.rand(point_cloud.shape[0], 1, 3, device=device) - 0.5) * 0.4 + 1
        # point_cloud[:,:,:3] = point_cloud[:,:,:3] / max_distance

        # if is_rotation is True:
        #     max_distance = torch.max(torch.sqrt(torch.sum((point_cloud[:,:,:3] - centroid) ** 2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0]
        #     point_cloud[:,:,:3] = point_cloud[:,:,:3] / max_distance
    return point_cloud

def centralize_pts(pc, is_rotation):
    with torch.no_grad():
        point_cloud = pc.clone()
        batch_size, n_points, _ = point_cloud.shape
        device = point_cloud.device  # Get the device from the input point cloud
        centroid = torch.mean(point_cloud[:,:,:3], dim=1, keepdim=True)
        point_cloud[:,:,:3] = point_cloud[:,:,:3] - centroid

        # if is_rotation is True:
        #     max_distance = torch.max(torch.sqrt(torch.sum((point_cloud[:,:,:3] - centroid) ** 2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0]
        #     point_cloud[:,:,:3] = point_cloud[:,:,:3] / max_distance
    return point_cloud

def random_rotation(pc):
    return pc

def train(is_rotation: bool,
          uniform: bool,
          use_normal: bool,
          num_epochs: int,
          root: str,
          num_points: int,
          lr: float,
          weight_decay: float,
          model_str: str,
          model_name: str,
          batch_size: int,
          accumulate: int,
          ckpt_dir: str):

    device = 'cuda'

    num_classes = 40
    data_path = os.path.join(root, "modelnet40_normal_resampled")
    train_dataset = ModelNetDataLoader(data_path, 
                            num_category=num_classes,
                            split='train', 
                            process_data=uniform,
                            transforms=False,
                            use_uniform_sample=uniform,
                            use_normals=use_normal,
                            num_point=num_points)

    test_dataset = ModelNetDataLoader(data_path, 
                            num_category=num_classes,
                            split='test', 
                            process_data=uniform,
                            transforms=is_rotation,
                            use_uniform_sample=uniform,
                            use_normals=use_normal,
                            num_point=num_points)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    model = parse_point_architecture(model_str, is_normal=use_normal, is_rotation=is_rotation)
    # model.is_ssl = False
    
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)
    
    # optimizer = Lamb(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), adam=True)
    # if is_rotation is True:
    if  True:
        optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=weight_decay
            )
        warmup_scheduler  = warmup.UntunedLinearWarmup(optimizer)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    num_steps         = num_epochs
    lr_scheduler      = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
    

    # max_gradient_norm = 1.0
    global_step       = 0
    start_epoch       = 0
    
    best_instance_acc = 0.0
    best_class_acc    = 0.0
    best_epoch        = 0
    
    step = 0
    
    log_file_name = f"training.log"
    
    print("[INFO] ... Starting the training phase ...")
    tik = time.time()
    for epoch in range(start_epoch, num_epochs):
        log_file = create_log_file(ckpt_dir, log_file_name)
        mean_correct      = []

        print('Epoch %d (%d/%s):' % (epoch + 1, epoch + 1, num_epochs))
        
        model = model.train()

        loop = tqdm(trainDataLoader)
        iter = 0
        first_loss = None

        for points, target in loop:
            
            step += 1
            
            if iter == len(trainDataLoader) - 1:
                break
            
            points, target = points.cuda(), target.cuda()
            points = points.to(torch.float32)
            augmented_points = augment(centralize_pts(points, is_rotation), is_rotation)
            real_points = centralize_pts(points, is_rotation)
            prob = torch.bernoulli(torch.ones(points.shape[0], 1, 1, device=points.device) * 0.5)
            points = augmented_points * prob + real_points * (1 - prob)
            target_oh = F.one_hot(target.long(), num_classes=40).to(points.dtype)
            pred, regularization = model(points)
            # pred = checkpoint_sequential(model, 4, points)
            
            loss = criterion(pred, target_oh) + regularization * 0.000
            
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loop.set_description("Train Accuracy: " + str(mean_correct[-1])[:5] + " Loss: " + str(loss.item())[:5])

            if first_loss == None:
                first_loss = loss.detach().item()

            loss = loss / accumulate
            loss.backward()# Plot the gradients

            if step % accumulate == 0:
                # torch.nn.utils.clip_grad_norm(model.parameters(), max_gradient_norm)
                optimizer.step()
                optimizer.zero_grad()

                # if is_rotation is True:
                if  True:
                    if iter < len(trainDataLoader) - 1:
                        with warmup_scheduler.dampening():
                            pass

            iter += 1
            global_step += 1
            
        train_instance_acc = np.mean(mean_correct)

        print("Train Instance Accuracy: " + str(train_instance_acc)[:7] + ". Loss Value: " + str(loss.detach().item())[:7] + ". First Loss: " + str(first_loss)[:7] + ". LR: " + str(round(lr_scheduler.get_last_lr()[0], 8)))

        log_file.write(f"Epoch {epoch+1}\n")
        log_file.write(f"Train Instance Accuracy={train_instance_acc:.5f}, Loss Value={loss.detach().item():.5f}, First Loss={first_loss:.5f}, Lr={lr_scheduler.get_last_lr()[0]:9f}\n")

        # if is_rotation is True:
        if  True:
            with warmup_scheduler.dampening():
                lr_scheduler.step()
                pass
        else:
            lr_scheduler.step()

        with torch.no_grad():
            test_mean_correct = []
            model = model.eval()
            test_class_acc = np.zeros((num_classes, 3))
            loop = tqdm(testDataLoader)
            for points, target in loop:
                points, target = points.cuda(), target.cuda()
                points = points.to(torch.float32)
                points = centralize_pts(points, is_rotation)
                if is_rotation is True:
                    points = random_rotation(points)
                pred = model(points)
                pred_choice = pred.data.max(1)[1]
                for cat in np.unique(target.cpu()):
                    classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                    test_class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
                    test_class_acc[cat, 1] += 1
                correct = pred_choice.eq(target.long().data).cpu().sum()
                test_mean_correct.append(correct.item() / float(points.size()[0]))
            test_class_acc[:, 2] =  test_class_acc[:, 0] / test_class_acc[:, 1]
            test_class_acc = np.mean(test_class_acc[:, 2])
            test_instance_acc = np.mean(test_mean_correct)

            if (test_instance_acc >= best_instance_acc):
                best_instance_acc = test_instance_acc
                best_epoch = epoch + 1
                print('Save best checkpoint model...')
                state = {
                    'epoch': epoch,
                    'train_instance_acc': train_instance_acc,
                    'test_instance_acc': test_instance_acc,
                    'test_class_acc': test_class_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, f"{ckpt_dir}/{model_name}_best.pth")
                
            if (test_class_acc >= best_class_acc):
                best_class_acc = test_class_acc

            print('Test Instance Accuracy: %.5f, Test Class Accuracy: %.5f'% (test_instance_acc, test_class_acc))
            print('Best_epoch: %i, Best Instance Accuracy: %f, Class Accuracy: %f'% (best_epoch, best_instance_acc, best_class_acc))

            log_file.write(f"Test Instance Accuracy={test_instance_acc:.5f}, Test Class Accuracy={test_class_acc:.5f}\n")
            log_file.write(f"Best Epoch={best_epoch}, Best Instance Accuracy={best_instance_acc:.5f}, Best Class Accuracy={best_class_acc:.5f}\n\n")
            
            # Keep checkpoint
            if (epoch + 1) == 200 or (epoch + 1) == num_epochs:
                state = {
                    'epoch': epoch,
                    'train_instance_acc': train_instance_acc,
                    'test_instance_acc': test_instance_acc,
                    'test_class_acc': test_class_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, f"{ckpt_dir}/{model_name}_epoch_{str(epoch+1)}.pth")
            else:
                state = {
                    'epoch': epoch,
                    'train_instance_acc': train_instance_acc,
                    'test_instance_acc': test_instance_acc,
                    'test_class_acc': test_class_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, f"{ckpt_dir}/{model_name}_latest.pth")
            print("\n")
            
    log_file.close()
    
    tok = time.time()
    print("[INFO] ... Training time: ", round((tok - tik) / 3600, 4), "Hrs")
        
if __name__ == '__main__':
    args = parse_args()
    experiment_settings = {}
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
        experiment_settings[arg] = value
    print("------------------------------------------------") 
    
    set_seed(args.seed)
    
    log_dir = os.path.join(
            args.checkpoint_dir, 
            args.exp_name, 
            f"seed_{args.seed}")
    
    os.makedirs(log_dir, exist_ok=True)

    file = open(f'{log_dir}/experiment_settings_{args.model_name}.txt', 'w')
    with file as f:
        for key, value in experiment_settings.items():  
            f.write('%s:%s\n' % (key, value))
            
    if args.uniform == 'True':
        args.uniform = True
    elif args.uniform == 'False':
        args.uniform = False
    else:
        raise ValueError
    
    if args.use_normal == 'True':
        args.use_normal = True
    elif args.use_normal == 'False':
        args.use_normal = False
    else:
        raise ValueError
    
    if args.is_rotation == 'True':
        args.is_rotation = True
    elif args.is_rotation == 'False':
        args.is_rotation = False
    else:
        raise ValueError
    
    train(uniform=args.uniform,
          num_epochs=args.num_epochs,
          root=args.root,
          use_normal=args.use_normal,
          num_points=args.num_points,
          lr=args.lr,
          weight_decay=args.weight_decay,
          model_str=args.model_str,
          model_name=args.model_name,
          batch_size=args.batch_size,
          accumulate=args.accumulate,
          ckpt_dir=log_dir,
          is_rotation=args.is_rotation)