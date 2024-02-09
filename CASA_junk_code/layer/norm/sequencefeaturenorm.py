import torch
import torch.nn as nn

class SequenceFeatureNorm(nn.Module):
    def __init__(self, num_features, epsilon=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        # Moving averages for mean and variance
        self.running_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.running_var = nn.Parameter(torch.zeros(num_features), requires_grad=False)

        self.calculated = False

    def forward(self, x, mask=None):
        if self.training:
            sumx = torch.sum(x, dim=1, keepdim=True)
            if mask is not None:
                x = x * (1 - mask.unsqueeze(-1))
                nx = torch.sum(mask, dim=1).unsqueeze(-1)
                meanx = sumx / nx
                varx = ((x - meanx) ** 2) / nx
                meanx = torch.mean(meanx, dim=[0, 2], keepdim=True)
                varx = torch.mean(varx, dim=0, keepdim=True)
            else:
                nx = x.shape[1]
                meanx = sumx / nx
                varx = torch.sum(((x - meanx) ** 2) / nx, dim=1, keepdim=True)
                meanx = torch.mean(meanx, dim=[0, 2], keepdim=True)
                varx = torch.mean(varx, dim=0, keepdim=True)

            if self.calculated == True:
                # Update running mean and variance with momentum
                self.running_mean.data = (1 - self.momentum) * self.running_mean + self.momentum * meanx
                self.running_var.data = (1 - self.momentum) * self.running_var + self.momentum * varx
            else:
                self.running_mean.data = meanx
                self.running_var.data = varx
                self.calculated = True

        x_normalized = (x - self.running_mean) / (torch.sqrt(self.running_var + self.epsilon) + self.epsilon)
        return x_normalized