from ..name import metrics_name_dict

import torch

import torch.nn as nn
import torch.nn.functional as F

class MeanSquareError(nn.Module):
    def __init__(self):
        """
            Brief description:
                Implementation of the Cross Entropy Loss function, which is a common loss function for classification problems.
                The loss function is defined as -sum(y * log(y_hat)) over all classes, where y is the true distribution and y_hat is the predicted distribution.

            Hyperparameters:
                n_classes (int): the number of target classes.

            Attributes:
                settings (dict): a dictionary that stores the name of the loss function and its hyperparameters.
                classes (int): the number of target classes.
        """
        super().__init__()        
        self.settings = {
            "name": "MSE",
        }
        self.differentiable = True

    def forward(self, inputs, targets, **kwargs):
        return F.mse_loss(inputs, targets)

class SmoothL1Loss(nn.Module):
    def __init__(self):
        """
            Brief description:
                Implementation of the Cross Entropy Loss function, which is a common loss function for classification problems.
                The loss function is defined as -sum(y * log(y_hat)) over all classes, where y is the true distribution and y_hat is the predicted distribution.

            Hyperparameters:
                n_classes (int): the number of target classes.

            Attributes:
                settings (dict): a dictionary that stores the name of the loss function and its hyperparameters.
                classes (int): the number of target classes.
        """
        super().__init__()        
        self.settings = {
            "name": "HUBER",
        }
        self.differentiable = True

    def forward(self, inputs, targets, **kwargs):
        return F.smooth_l1_loss(inputs, targets)

metrics_name_dict["MSE"] = MeanSquareError
metrics_name_dict["HUBER"] = SmoothL1Loss
