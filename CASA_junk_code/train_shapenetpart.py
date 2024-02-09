import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
import pytorch_warmup as warmup
from pytorch_lamb import Lamb, log_lamb_rs
import numpy as np
import random
import time
from tqdm import tqdm
import argparse
from model import parse_point_architecture
from data.dataset import PartNormalDataset
import warnings
warnings.filterwarnings('ignore')


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

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


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
    parser.add_argument("--num_classes", type=int, default=50, help="Number of training classes")
    parser.add_argument("--model_str", type=str, help="Architecture, example: PCLL,96@RESC,96,96,8,192,0.1,1,True@RESC,96,96,8,192,0.1,1,True@RESC,96,96,8,192,0.1,1,True@CLFH,40")
    parser.add_argument("--model_name", type=str, help="Name to save model.")
    parser.add_argument("--is_rotation", type=str, default='True', help="Name to save model.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, required=False, help="Weight decay value of optimize function")
    parser.add_argument("--seed", type=int, default=1, help="Training seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path to training checkpoints")
    parser.add_argument("--exp_name", type=str, default="exp", help="Experiment name")
    parser.add_argument("--use_normal", type=str, help=".")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of workers for loading data")
    parser.add_argument("--accumulate", type=int, default=4, help=".")
    
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
        # max_distance = torch.max(torch.sqrt(torch.sum((point_cloud[:,:,:3] - centroid) ** 2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0]
        # point_cloud[:,:,:3] = point_cloud[:,:,:3] / max_distance

        # if is_rotation is True:
        #     max_distance = torch.max(torch.sqrt(torch.sum((point_cloud[:,:,:3] - centroid) ** 2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0]
        #     point_cloud[:,:,:3] = point_cloud[:,:,:3] / max_distance
    return point_cloud

def train(use_normal: bool,
          is_rotation: bool,
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

    train_dataset = PartNormalDataset(root = './datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                                      npoints=2500, 
                                      split='train', 
                                      class_choice=None, 
                                      normal_channel=use_normal)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size=batch_size, 
                                                  shuffle=True, 
                                                  num_workers=num_workers, 
                                                  drop_last=True)

    test_dataset = PartNormalDataset(root = './datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                                      npoints=2500, 
                                      split='test', 
                                      class_choice=None, 
                                      normal_channel=use_normal)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, 
                                                 batch_size=batch_size, 
                                                 shuffle=False, 
                                                 num_workers=num_workers)

    model = parse_point_architecture(model_str, is_normal=use_normal, is_rotation=is_rotation)
    
    # model.is_ssl = False
    # num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("[INFO]...Number of parameters of model: ", num_model_params)
    
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    # if is_rotation is True:
    if True:
        optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=weight_decay
            )
        warmup_scheduler  = warmup.UntunedLinearWarmup(optimizer)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # optimizer = Lamb(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), adam=True)sss


    num_steps         = num_epochs # len(trainDataLoader)
    # lr_scheduler      = CosineAnnealingLR(optimizer, T_max=num_steps)
    lr_scheduler      = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
    
    global_step       = 0
    start_epoch       = 0
    # max_gradient_norm = 5.0

    best_epoch        = 0
    best_acc = 0
    best_class_avg_iou = 0
    best_instance_avg_iou = 0
    
    step = 0
    
    log_file_name = f"training.log"
    
    print("[INFO] ... Starting the training phase ...")
    tik = time.time()
    for epoch in range(start_epoch, num_epochs):
        log_file = create_log_file(ckpt_dir, log_file_name)
        
        train_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_classes)]
        total_correct_class = [0 for _ in range(num_classes)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat
                
        print('Epoch %d (%d/%s):' % (epoch + 1, epoch + 1, num_epochs))
        
        model = model.train()

        loop = tqdm(trainDataLoader)
        iter = 0
        first_loss = None

        for points, _, target in loop:
            
            step += 1
            
            if iter == len(trainDataLoader) - 1:
                break
            
            cur_batch_size, NUM_POINT, _ = points.size()
            points, target = points.cuda(), target.cuda()
            augmented_points = augment(centralize_pts(points, is_rotation), is_rotation)
            real_points = centralize_pts(points, is_rotation)
            prob = torch.bernoulli(torch.ones(points.shape[0], 1, 1, device=points.device) * 0.5)
            points = augmented_points * prob + real_points * (1 - prob)
            # optimizer.zero_grad()
            
            pred, regularization = model(points)

            # Get loss
            # pred = pred.reshape(-1, num_classes)
            # target = target.reshape(-1)
            loss = criterion(pred.reshape(-1, num_classes), target.reshape(-1).long()) + regularization * 0.000
            loop.set_description(str(loss.item())[:8])
            
            # Get score
            cur_pred_val = pred.cpu().data.numpy()
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

            for l in range(num_classes):
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
            
            if first_loss == None:
                first_loss = loss.detach().item()
            
            loss = loss / accumulate
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm(model.parameters(), max_gradient_norm)
            # optimizer.step()
            
            
            # if iter < len(trainDataLoader) - 1:
            #     with warmup_scheduler.dampening():
            #         pass
            # iter += 1
            # global_step += 1
            
            if step % accumulate == 0:
                # torch.nn.utils.clip_grad_norm(model.parameters(), max_gradient_norm)
                optimizer.step()
                optimizer.zero_grad()

                # if is_rotation is True:
                if True:
                    if iter < len(trainDataLoader) - 1:
                        with warmup_scheduler.dampening():
                            pass
            iter += 1
            global_step += 1

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        train_metrics['accuracy'] = total_correct / float(total_seen)
        train_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
        for cat in sorted(shape_ious.keys()):
            print('Train mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            log_file.write('Train mIoU of ' + str((cat + ' ' * (14 - len(cat)))) + ' '+ str(shape_ious[cat]) + '\n')
        log_file.write('\n\n')
        train_metrics['class_avg_iou'] = mean_shape_ious
        train_metrics['instance_avg_iou'] = np.mean(all_shape_ious)
        
        print('Train Loss is: %5f' % loss.detach().item(), "LR: ", round(lr_scheduler.get_last_lr()[0], 8))
        print('Train Accuracy is: %.5f' % train_metrics['accuracy'])
        print('Train Class avg mIOU is: %.5f' % train_metrics['class_avg_iou'])
        print('Train Instance avg mIOU is: %.5f' % train_metrics['instance_avg_iou'])
        
        log_file.write(f"Epoch {epoch+1}\n")
        log_file.write(f"Train Loss={loss.detach().item():.5f}, Lr={lr_scheduler.get_last_lr()[0]:9f}\n")
        log_file.write(f"Train Accuracy={train_metrics['accuracy']:.5f}, Train Instance avg mIOU={train_metrics['instance_avg_iou']:5f}, Train Class avg mIOU={train_metrics['class_avg_iou']:5f}\n")
        
        if is_rotation is True:
            with warmup_scheduler.dampening():
                lr_scheduler.step()
        else:
            lr_scheduler.step()
    

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_classes)]
            total_correct_class = [0 for _ in range(num_classes)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            model = model.eval()

            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                points = centralize_pts(points, is_rotation)
                seg_pred = model(points)

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

                for l in range(num_classes):
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
            for cat in sorted(shape_ious.keys()):
                print('Eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
                log_file.write('Eval mIoU of ' + str((cat + ' ' * (14 - len(cat)))) + ' '+ str(shape_ious[cat]) + '\n')
            log_file.write('\n\n')
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)
            
            
            print('Current epoch is: %i' % (epoch + 1))
            print('Test accuracy is: %.5f' % test_metrics['accuracy'])
            print('Test class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])
            print('Test instance avg mIOU is: %.5f' % test_metrics['instance_avg_iou'])
            
            if (test_metrics['class_avg_iou'] >= best_class_avg_iou):
                print('Save best checkpoint model...')
                state = {
                'epoch': epoch,
                'train_acc': train_metrics['accuracy'],
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'instance_avg_iou': test_metrics['instance_avg_iou'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, f"{ckpt_dir}/{name}_best.pth")

            if (test_metrics['instance_avg_iou'] >= best_instance_avg_iou):
                print('Save best checkpoint model...')
                state = {
                'epoch': epoch,
                'train_acc': train_metrics['accuracy'],
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'instance_avg_iou': test_metrics['instance_avg_iou'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, f"{ckpt_dir}/{name}_best_instance.pth")
                
            if test_metrics['accuracy'] > best_acc:
                best_acc = test_metrics['accuracy']
                
            if test_metrics['class_avg_iou'] > best_class_avg_iou:
                best_class_avg_iou = test_metrics['class_avg_iou']
                best_epoch = epoch + 1
                
            if test_metrics['instance_avg_iou'] > best_instance_avg_iou:
                best_instance_avg_iou = test_metrics['instance_avg_iou']
                best_epoch = epoch + 1

            print('Best epoch is: %i' % best_epoch)
            print('Best accuracy is: %.5f' % best_acc)
            print('Best class avg mIOU is: %.5f' % best_class_avg_iou)
            print('Best instance avg mIOU is: %.5f' % best_instance_avg_iou)
            
            
            log_file.write(f"Test Accuracy={test_metrics['accuracy']:.5f}, Test Instance avg mIOU={test_metrics['instance_avg_iou']:5f}, Test Class avg mIOU={test_metrics['class_avg_iou']:5f}\n")
            log_file.write(f"Best Epoch={best_epoch}, Best Accuracy={best_acc:.5f}, Best Instance avg mIOU={best_instance_avg_iou:5f}, Best Class avg mIOU={best_class_avg_iou:5f}\n\n")

            # Keep checkpoint
            if (epoch + 1) == 200 or (epoch + 1) == num_epochs:
                state = {
                'epoch': epoch,
                'train_acc': train_metrics['accuracy'],
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'instance_avg_iou': test_metrics['instance_avg_iou'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, f"{ckpt_dir}/{name}_epoch_{str(epoch+1)}.pth")
            else:
                state = {
                'epoch': epoch,
                'train_acc': train_metrics['accuracy'],
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'instance_avg_iou': test_metrics['instance_avg_iou'],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
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
    
    train(num_epochs=args.num_epochs,
          num_classes=args.num_classes,
          is_rotation=args.is_rotation,
          use_normal=args.use_normal,
          lr=args.lr,
          weight_decay=args.weight_decay,
          model_str=args.model_str,
          name=args.model_name,
          batch_size=args.batch_size,
          ckpt_dir=log_dir,
          accumulate=args.accumulate,
          num_workers=args.num_workers)
