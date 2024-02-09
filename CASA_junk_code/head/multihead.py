import torch.nn as nn

class Head(nn.Module):
    def __init__(self, tasks):
        super().__init__()
        self.tasks = nn.ModuleList(tasks)

    def forward(self, x, task=None):
        if task == None:
            return [self.tasks[i](x) for i in range(len(self.tasks))]
        return self.tasks[task](x)