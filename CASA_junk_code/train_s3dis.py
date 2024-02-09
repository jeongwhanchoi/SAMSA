import os
import torch
import datetime
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import pytorch_warmup as warmup
from pytorch_lamb import Lamb, log_lamb_rs
import numpy as np
import random
import time
from tqdm import tqdm
import argparse
from model import parse_point_architecture
from data.dataset import S3DISDataset
import warnings
warnings.filterwarnings('ignore')

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
        
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
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--num_classes", type=int, default=13, help="Number of training classes")
    parser.add_argument("--model_str", type=str, help="Architecture, example: PCLL,96@RESC,96,96,8,192,0.1,1,True@RESC,96,96,8,192,0.1,1,True@RESC,96,96,8,192,0.1,1,True@CLFH,40")
    parser.add_argument("--model_name", type=str, help="Name to save model.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, required=False, help="Weight decay value of optimize function")
    parser.add_argument("--seed", type=int, default=1, help="Training seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path to training checkpoints")
    parser.add_argument("--exp_name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--num_workers", type=int, default=14, help="Number of workers for loading data")
    parser.add_argument("--accumulate", type=int, default=4, help=".")
    
    return parser.parse_args()


def train(
          num_epochs: int,
          num_classes: int,
          lr: float,
          weight_decay: float,
          model_str: str,
          name: str,
          batch_size: int,
          ckpt_dir: str,
          accumulate: int,
          num_workers: int):
    
    device = 'cuda'
    
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 1.0

    train_dataset = S3DISDataset(split='train', 
                                 data_root='./datasets/stanford_indoor3d', 
                                 num_point=num_point, 
                                 test_area=test_area, 
                                 block_size=block_size, 
                                 sample_rate=sample_rate, 
                                 transform=None)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                               num_workers=num_workers, 
                                               pin_memory=True,
                                               drop_last=True)

    test_dataset = S3DISDataset(split='test', 
                                 data_root='./datasets/stanford_indoor3d', 
                                 num_point=num_point, 
                                 test_area=test_area, 
                                 block_size=block_size, 
                                 sample_rate=sample_rate, 
                                 transform=None)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, 
                                                 batch_size=batch_size, 
                                                 shuffle=False, 
                                                 num_workers=num_workers,
                                                 pin_memory=True,
                                                 drop_last=True)
    weights = torch.Tensor(train_dataset.labelweights).cuda()
    
    model = parse_point_architecture(model_str)
    
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = model.to(device)
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay
        )
    
    # optimizer = Lamb(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), adam=True)

    lr_scheduler      = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=50, gamma=0.9)

    warmup_scheduler  = warmup.UntunedLinearWarmup(optimizer)
    global_step       = 0
    start_epoch       = 0
    max_gradient_norm = 1.0

    best_epoch        = 0
    best_iou = 0
    step = 0
    
    time_now = datetime.datetime.now()
    log_file_name = f"training_{str(time_now)}.log"
    
    print("[INFO] ... Starting the training phase ...")
    tik = time.time()
    for epoch in range(start_epoch, num_epochs):
        log_file = create_log_file(ckpt_dir, log_file_name)
        
        train_metrics = {}
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        
        labelweights = np.zeros(num_classes)
        total_seen_class = [0 for _ in range(num_classes)]
        total_correct_class = [0 for _ in range(num_classes)]
        total_iou_deno_class = [0 for _ in range(num_classes)]
             
        print('Epoch %d (%d/%s):' % (epoch + 1, epoch + 1, num_epochs))
        
        model = model.train()

        loop = tqdm(trainDataLoader)
        num_batches = len(trainDataLoader)
        iter = 0
        first_loss = None

        for points, target in loop:
            step += 1
            if iter == len(trainDataLoader) - 1:
                break
            
            _, NUM_POINT, _ = points.size()
            points, target = points.float().cuda(), target.long().cuda()
            # optimizer.zero_grad()
            
            pred = model(points)

            # Get loss
            loss = criterion(pred.reshape(-1, num_classes), target.reshape(-1).long())
            
            if first_loss == None:
                first_loss = loss.detach().item()
            
            loss = loss / accumulate
            loss_sum += loss
            
            loss.backward()
            
            if step % accumulate == 0:
                # torch.nn.utils.clip_grad_norm(model.parameters(), max_gradient_norm)
                optimizer.step()
                optimizer.zero_grad()
            
                if iter < len(trainDataLoader) - 1:
                    with warmup_scheduler.dampening():
                        pass
            iter += 1
            global_step += 1
                        
            loop.set_description(str(loss.item())[:8])
            
            pred = pred.contiguous().view(-1, num_classes)
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            pred_choice = pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (batch_size * NUM_POINT)

            tmp, _ = np.histogram(batch_label, range(num_classes + 1))
            labelweights += tmp

            for l in range(num_classes):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l] += np.sum((pred_choice == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_choice == l) | (batch_label == l)))
                                
        labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
        
        train_metrics['lr'] = round(lr_scheduler.get_last_lr()[0], 8)
        train_metrics['loss'] = loss_sum / float(num_batches)
        train_metrics['accuracy'] = total_correct / float(total_seen)
        train_metrics['mIoU'] = mIoU
        train_metrics['class_accuracy'] = np.mean(
            np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))
        
        print('Learning rate: %f' % (train_metrics['lr']))
        print('Train mean loss: %f' % (train_metrics['loss']))
        print('Train accuracy: %.5f' % train_metrics['accuracy'])
        print('Train avg class IoU: %.5f' % (train_metrics['mIoU']))
        print('Train avg class Acc: %.5f' % (train_metrics['class_accuracy']))

        iou_per_class_str = '------- IoU --------\n'
        for l in range(num_classes):
            iou_per_class_str += 'class %s weight: %.5f, IoU: %.5f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                total_correct_class[l] / float(total_iou_deno_class[l]))
        print(iou_per_class_str)
        
        log_file.write(f"Epoch={epoch+1}\n")
        log_file.write(f"Learning rate={train_metrics['lr']}\n")
        log_file.write(f"Train Loss={train_metrics['loss']}\n")
        log_file.write(f"Train Acc={train_metrics['accuracy']:.5f}, Train avg class mIOU={train_metrics['mIoU']:5f}, Train avg class Acc={train_metrics['class_accuracy']:5f}\n")
        
        with warmup_scheduler.dampening():
            lr_scheduler.step()
    
        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(num_classes)
            total_seen_class = [0 for _ in range(num_classes)]
            total_correct_class = [0 for _ in range(num_classes)]
            total_iou_deno_class = [0 for _ in range(num_classes)]

            model = model.eval()

            for _, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                _, NUM_POINT, _ = points.size()
                points, target = points.float().cuda(), target.long().cuda()
                pred = model(points)
                
                loss = criterion(pred.reshape(-1, num_classes), target.reshape(-1).long())
                loss_sum += loss

                pred_val = pred.contiguous().cpu().data.numpy()
                
                batch_label = target.cpu().data.numpy()

                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (batch_size * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(num_classes + 1))
                labelweights += tmp

                for l in range(num_classes):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
                                   
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
            
            test_metrics['loss'] = loss_sum / float(num_batches)
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['mIoU'] = mIoU
            test_metrics['class_accuracy'] = np.mean(
                np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))
            
            print('Test mean loss: %f' % (test_metrics['loss']))
            print('Test accuracy: %.5f' % test_metrics['accuracy'])
            print('Test avg class IoU: %.5f' % (test_metrics['mIoU']))
            print('Test avg class Acc: %.5f' % (test_metrics['class_accuracy']))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(num_classes):
                iou_per_class_str += 'class %s weight: %.5f, IoU: %.5f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))
            print(iou_per_class_str)
            
            if (test_metrics['mIoU'] >= best_iou):
                best_epoch = epoch + 1
                best_iou = test_metrics['mIoU']
                print('Save best mIoU checkpoint model...')
                state = {
                'epoch': epoch+1,
                'train_acc': train_metrics['accuracy'],
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['mIoU'],
                'class_avg_acc': test_metrics['class_accuracy'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'warmup_state_dict': warmup_scheduler.state_dict(),
                }
                torch.save(state, f"{ckpt_dir}/{name}_best.pth")

            log_file.write(f"Test Loss={test_metrics['loss']}\n")
            log_file.write(iou_per_class_str + '\n')
            log_file.write(f"Test Acc={test_metrics['accuracy']:.5f}, Test avg class mIOU={test_metrics['mIoU']:5f}, Test avg class Acc={test_metrics['class_accuracy']:5f}\n")
 
            print('Best epoch is: %i' % best_epoch)
            print('Best class mIoU is: %.5f' % best_iou)
        
            log_file.write(f"Best Epoch={best_epoch}, Best avg class IoU={best_iou:.5f}\n\n")

            # Keep checkpoint
            if (epoch + 1) == 5 or (epoch + 1) == num_epochs:
                state = {
                'epoch': epoch+1,
                'train_acc': train_metrics['accuracy'],
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['mIoU'],
                'class_avg_acc': test_metrics['class_accuracy'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'warmup_state_dict': warmup_scheduler.state_dict(),
                }
                torch.save(state, f"{ckpt_dir}/{name}_epoch_{str(epoch+1)}.pth")
            else:
                state = {
                'epoch': epoch+1,
                'train_acc': train_metrics['accuracy'],
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['mIoU'],
                'class_avg_acc': test_metrics['class_accuracy'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'warmup_state_dict': warmup_scheduler.state_dict(),
                }
                torch.save(state, f"{ckpt_dir}/{name}_latest.pth")
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
          num_classes=args.num_classes,
          lr=args.lr,
          weight_decay=args.weight_decay,
          model_str=args.model_str,
          name=args.model_name,
          batch_size=args.batch_size,
          ckpt_dir=log_dir,
          accumulate=args.accumulate,
          num_workers=args.num_workers)
