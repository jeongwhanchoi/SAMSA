import torch
import torch.nn as nn

class BatchNormMean(nn.Module):
    def __init__(self, num_features, epsilon=1e-5, momentum=0.1):
        super(BatchNormMean, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        # Moving averages for mean and variance
        self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)

        self.calculated = False

    def forward(self, x):
        if self.training:
            mean = torch.mean(x, dim=0, keepdim=True)[0]
            mean = torch.mean(mean, dim=[i for i in range(len(mean.shape) - 1)], keepdim=True)

            if self.calculated == True:
                # Update running mean and variance with momentum
                self.running_mean.data = (1 - self.momentum) * self.running_mean + self.momentum * mean
            else:
                self.running_mean.data = mean
                self.calculated = True

        x_normalized = x / (self.running_mean + self.epsilon)
        return x_normalized