from ..name import module_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomScale(nn.Module):
    def __init__(self, 
                 scale_min: float = 0.6, 
                 scale_max: float = 1.4,
                 same_scale_for_each_dimension: bool = False,
                 test_time_augmentation: bool = True):
        """
            Random Scale Data Augmentation Module.
            Hyperparameters:
                scale_min (float)
                scale_max: (float)
                same_scale_for_each_dimension (bool): Whether to randomly scale different dimension differently 
                test_time_augmentation (bool): Whether to use test time data augmentation
        """
        super().__init__()
        self.settings = {
            "name": "RANDSCALE",
            "scale_min": scale_min,
            "scale_max": scale_max,
            "same_scale_for_each_dimension": same_scale_for_each_dimension
        }
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.range = self.scale_max - self.scale_min
        if same_scale_for_each_dimension is True:
            self.forward = self.forward_same_scale
        else:
            self.forward = self.forward_different_scale
        self.test_time_augmentation = test_time_augmentation

    def forward_different_scale(self, x, mask=None, **args):
        if self.training is True or self.test_time_augmentation is True:
            scale = torch.rand(x.shape[0], x.shape[1], 3, device=x.device) * self.range + self.scale_min
            x[:,:,:3] = x[:,:,:3] * scale
            return {'x': x, 'x_coords': x[:,:,:3]}
        return {'x': x, 'x_coords': x[:,:,:3]}
    
    def forward_same_scale(self, x, mask=None, **args):
        if self.training is True or self.test_time_augmentation is True:
            scale = torch.randn(x.shape[0], x.shape[1], 1, device=x.device) * self.range + self.scale_min
            x[:,:,:3] = x[:,:,:3] * scale
            return {'x': x, 'x_coords': x[:,:,:3]}
        return {'x': x, 'x_coords': x[:,:,:3]}
    
module_name_dict["RANDSCALE"] = RandomScale

class RandomRotateZ(nn.Module):
    def __init__(self, p=0.5, test_time_augmentation: bool = True):
        """
            Random Rotation around Z axis Data Augmentation Module.
            Hyperparameters:
                test_time_augmentation (bool): Whether to use test time data augmentation
        """
        super().__init__()
        self.settings = {
            "name": "RANDROTATEZ"
        }
        self.test_time_augmentation = test_time_augmentation
        self.p = p

    def forward(self, x, mask=None, **args):
        if self.training is True or self.test_time_augmentation is True:
            p = torch.heaviside(torch.rand(x.shape[0], 1, 1, device=x.device) - self.p, torch.zeros(1, device=x.device))
            theta = torch.rand(x.shape[0], device=x.device) * 2 * torch.pi
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotation_matrix = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=-1).reshape(-1, 2, 2)
            copy_x = x.clone()
            x[:, :, :2] = torch.einsum('bij,bnj->bni', rotation_matrix, x[:, :, :2])
            x = x * (1 - p) + copy_x * p
            return {'x': x, 'x_coords': x[:,:,:3]}
        return {'x': x, 'x_coords': x[:,:,:3]}
    
module_name_dict["RANDROTATEZ"] = RandomRotateZ


class RandomJitter(nn.Module):
    def __init__(self, sigma: float = 0.01, clip: float = 0.05, test_time_augmentation: bool = True):
        """
            Random Jitter Data Augmentation Module.
            Hyperparameters:
                sigma (float): Standard deviation of jitter noise
                clip (float): Clipping value for jitter noise
                test_time_augmentation (bool): Whether to use test time data augmentation
        """
        super().__init__()
        self.settings = {
            "name": "RANDJITTER",
            "sigma": sigma,
            "clip": clip
        }
        self.sigma = sigma
        self.clip = clip
        self.test_time_augmentation = test_time_augmentation

    def forward(self, x, mask=None, **args):
        if self.training is True or self.test_time_augmentation is True:
            noise = torch.clamp(self.sigma * torch.randn_like(x), -self.clip, self.clip)
            x = x + noise
            return {'x': x, 'x_coords': x[:,:,:3]}
        return {'x': x, 'x_coords': x[:,:,:3]}
    
module_name_dict["RANDJITTER"] = RandomJitter


class RandomFlipXY(nn.Module):
    def __init__(self, prob: float = 0.5, test_time_augmentation: bool = True):
        """
            Random Flip for X/Y dimensions Data Augmentation Module.
            Hyperparameters:
                prob (float): Probability of flipping
                test_time_augmentation (bool): Whether to use test time data augmentation
        """
        super().__init__()
        self.settings = {
            "name": "RANDFLIPXY",
            "prob": prob
        }
        self.prob = prob
        self.test_time_augmentation = test_time_augmentation

    def forward(self, x, x_coords, mask=None, **args):
        with torch.no_grad():
            if self.training is True or self.test_time_augmentation is True:
                x_flip = x_coords[:, :, [1, 0, 2]]
                p = torch.heaviside(torch.randn(x_coords.shape[0], 1, 1, device=x.device), torch.zeros(1, device=x.device))
                x_coords = (x_coords * p + x_flip * (1 - p))
                x[:, :, :3] = x_coords
                return {'x': x, 'x_coords': x_coords}
            return {'x': x, 'x_coords': x_coords}
    
module_name_dict["RANDFLIPXY"] = RandomFlipXY

class Centralize(nn.Module):
    def __init__(self):
        """
            Centroid to (0,0,0)
        """
        super().__init__()
        self.settings = {
            "name": "CENTRALIZE"
        }

    def forward(self, x, x_coords, mask=None, **args):
        with torch.no_grad():
            mean_x_coords = torch.sum(x_coords, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True).unsqueeze(-1)
            x[:, :, :3] = x_coords
            return {'x': x, 'x_coords': x_coords}
    
module_name_dict["CENTRALIZE"] = Centralize

class DropPoint(nn.Module):
    def __init__(self, minp: float = 0.5, p: float = 0.5, test_time_augmentation: bool = True):
        """
            Random Flip for X/Y dimensions Data Augmentation Module.
            Hyperparameters:
                prob (float): Probability of flipping
                test_time_augmentation (bool): Whether to use test time data augmentation
        """
        super().__init__()
        self.settings = {
            "name": "DROPPOINT",
            "minp": minp,
            "prob": p
        }
        self.prob = p
        self.minp = minp
        self.test_time_augmentation = test_time_augmentation

    def forward(self, x, x_coords, mask=None, **args):
        with torch.no_grad():
            if self.training is True or self.test_time_augmentation is True:
                p_drop = torch.rand(x.shape[0], 1, device=x.device) * (1 - self.minp) + self.minp
                mask_drop = torch.heaviside(p_drop - torch.rand(x.shape[0], x.shape[1], device=x.device), values=torch.zeros(1, device=x.device)) * mask
                p = torch.heaviside(torch.randn(x_coords.shape[0], 1, device=x.device), torch.zeros(1, device=x.device))
                mask = (mask * p + mask_drop * (1 - p))
                return {'mask': mask}
            return {'mask': mask}

module_name_dict["DROPPOINT"] = DropPoint