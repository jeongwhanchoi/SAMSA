import torch
import torch.nn as nn

class BatchNormNear(nn.Module):
    def __init__(self, num_features, epsilon=1e-5, momentum=0.1):
        super(BatchNormNear, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        # Moving averages for mean and variance
        self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)

        self.calculated = False

    def forward(self, x):
        if self.training:
            mean = torch.min(x, dim=0, keepdim=True)[0]
            mean = torch.mean(mean, dim=[i for i in range(len(mean.shape) - 1)], keepdim=True)

            if self.calculated == True:
                # Update running mean and variance with momentum
                self.running_mean.data = (1 - self.momentum) * self.running_mean + self.momentum * mean
            else:
                self.running_mean.data = mean
                self.calculated = True

        x_normalized = (x - self.running_mean) / (self.running_mean + self.epsilon)
        return x_normalized
    
# class BatchNormNear(nn.Module):
#     def __init__(self, num_features, epsilon=1e-5, momentum=0.1):
#         super(BatchNormNear, self).__init__()
#         self.num_features = num_features
#         self.epsilon = epsilon
#         self.momentum = momentum

#         # Moving averages for mean and variance
#         self.running_min = nn.Parameter(torch.zeros(num_features), requires_grad=False)
#         self.running_max = nn.Parameter(torch.zeros(num_features), requires_grad=False)

#         self.calculated = False

#     def forward(self, x):
#         if self.training:
#             min_ = torch.min(x, dim=0, keepdim=True)[0]
#             min_ = torch.mean(min_, dim=[i for i in range(len(min_.shape) - 1)], keepdim=True)

#             max_ = torch.max(x, dim=0, keepdim=True)[0]
#             max_ = torch.mean(max_, dim=[i for i in range(len(max_.shape) - 1)], keepdim=True)

#             if self.calculated == True:
#                 # Update running mean and variance with momentum
#                 self.running_min.data = (1 - self.momentum) * self.running_min + self.momentum * min_
#                 self.running_max.data = (1 - self.momentum) * self.running_max + self.momentum * max_
#             else:
#                 self.running_min.data = min_
#                 self.running_max.data = max_
#                 self.calculated = True

#         x_normalized = (x - self.running_min) / (self.running_max + self.epsilon)
#         return x_normalized