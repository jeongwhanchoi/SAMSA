from ..name import metrics_name_dict

import torch

import torch.nn as nn
import torch.nn.functional as F

class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()        
        self.settings = {
            "name": "BINCENTROPY",
        }
        self.differentiable = True

    def forward(self, inputs, targets, **kwargs):
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        return ce_loss
    
metrics_name_dict['BINCENTROPY'] = BinaryCrossEntropy

class CrossEntropy(nn.Module):
    def __init__(self, n_classes):
        """
            Brief description:
                Implementation of the Cross Entropy Loss function, which is a common loss function for classification problems.

            Hyperparameters:
                n_classes (int): the number of target classes.
        """
        super().__init__()        
        self.settings = {
            "name": "CENTROPY",
            "n_classes": n_classes,
        }
        self.classes = n_classes

    def forward(self, inputs, targets, **kwargs):
        inputs = inputs.reshape(-1, self.classes)
        targets = targets.reshape(-1).to(torch.long)
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.classes)
        return ce_loss
    
metrics_name_dict['CENTROPY'] = CrossEntropy

class BalancedCrossEntropy(nn.Module):
    def __init__(self, n_classes):
        """
            Brief description:
                Implementation of the Balanced Cross Entropy Loss function, which is a CrossEntropy with weight rebalancing.

            Hyperparameters:
                n_classes (int): the number of target classes.
        """
        super().__init__()        
        self.settings = {
            "name": "BCENTROPY",
            "n_classes": n_classes,
        }
        self.classes = n_classes

    def forward(self, inputs, targets, weight=None, **kwargs):
        inputs = inputs.reshape(-1, self.classes)
        targets = targets.reshape(-1).to(torch.long)
        if weight is None:
            ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.classes)
        else:
            ce_loss = F.cross_entropy(inputs, targets, weight=weight, ignore_index=self.classes)
        return ce_loss
    
metrics_name_dict['BCENTROPY'] = BalancedCrossEntropy