### IMPORT LIBRARIES

## AUTOMATIC DIFFERENTIATION
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_warmup as warmup
import numpy as np
import random
import time
from copy import deepcopy
from tqdm import tqdm
from multimethod import multimethod
import argparse
import logging
from model import parse_sequence_architecture
from data.dataset import ListOPS
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
    parser.add_argument("--root", type=str, default="./datasets/LRA/", help="Dataset directory folder.")
    parser.add_argument("--model_str", type=str, help="Architecture, example: PCLL,96@RESC,96,96,8,192,0.1,1,True@RESC,96,96,8,192,0.1,1,True@RESC,96,96,8,192,0.1,1,True@CLFH,40")
    parser.add_argument("--model_name", type=str, help="Name to save model.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, required=False, help="Weight decay value of optimize function")
    parser.add_argument("--seed", type=int, default=1, help="Training seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path to training checkpoints")
    parser.add_argument("--exp_name", type=str, default="wo_fps_depth_3", help="Experiment name")
    parser.add_argument("--accumulate", type=int, help=".")
    
    return parser.parse_args()

def train(num_epochs: int,
          root: str,
          lr: float,
          weight_decay: float,
          model_str: str,
          model_name: str,
          batch_size: int,
          accumulate: int,
          ckpt_dir: str):

    device = 'cuda'

    num_classes = 10
    # data_path = root + "\\data\\LRA\\"
    train_dataset = ListOPS(split='train', path_folder=root)
    test_dataset = ListOPS(split='test', path_folder=root)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    model = parse_sequence_architecture(model_str, 2048)
    
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)
    
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay
        )
    
    num_steps         = num_epochs
    lr_scheduler      = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    
    warmup_scheduler  = warmup.UntunedLinearWarmup(optimizer)
    max_gradient_norm = 1.0
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

        for input1, target in loop:
            
            step += 1
            
            if iter == len(trainDataLoader) - 1:
                break
            
            input1, target = input1.cuda(), target.cuda()
            target_oh = F.one_hot(target.long(), num_classes=num_classes).to(input1.dtype)
            pred = model(input1)
            
            loss = criterion(pred, target_oh)
            
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(input1.size()[0]))
            loop.set_description("Train Accuracy: " + str(mean_correct[-1])[:5] + " Loss: " + str(loss.item())[:5])

            if first_loss == None:
                first_loss = loss.detach().item()

            loss = loss / accumulate
            loss.backward()

            if step % accumulate == 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), max_gradient_norm)
                optimizer.step()
                optimizer.zero_grad()
            
                if iter < len(trainDataLoader) - 1:
                    with warmup_scheduler.dampening():
                        pass
            iter += 1
            global_step += 1
            
        train_instance_acc = np.mean(mean_correct)

        print("Train Instance Accuracy: " + str(train_instance_acc)[:7] + ". Loss Value: " + str(loss.detach().item())[:7] + ". First Loss: " + str(first_loss)[:7] + ". LR: " + str(round(lr_scheduler.get_last_lr()[0], 8)))

        log_file.write(f"Epoch {epoch+1}\n")
        log_file.write(f"Train Instance Accuracy={train_instance_acc:.5f}, Loss Value={loss.detach().item():.5f}, First Loss={first_loss:.5f}, Lr={lr_scheduler.get_last_lr()[0]:9f}\n")

        with warmup_scheduler.dampening():
            lr_scheduler.step()

        with torch.no_grad():
            test_mean_correct = []
            model = model.eval()
            test_class_acc = np.zeros((num_classes, 3))
            loop = tqdm(testDataLoader)
            for input1, target in loop:
                input1, target = input1.cuda(), target.cuda()
                pred = model(input1)
                pred_choice = pred.data.max(1)[1]
                for cat in np.unique(target.cpu()):
                    classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                    test_class_acc[cat, 0] += classacc.item() / float(input1[target == cat].size()[0])
                    test_class_acc[cat, 1] += 1
                correct = pred_choice.eq(target.long().data).cpu().sum()
                test_mean_correct.append(correct.item() / float(input1.size()[0]))
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
    
    train(num_epochs=args.num_epochs,
          root=args.root,
          lr=args.lr,
          weight_decay=args.weight_decay,
          model_str=args.model_str,
          model_name=args.model_name,
          batch_size=args.batch_size,
          accumulate=args.accumulate,
          ckpt_dir=log_dir)