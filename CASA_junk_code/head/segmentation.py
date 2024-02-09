import torch.nn as nn
    
class SegmentationHead(nn.Module):
    def __init__(self,
                 n_classes: int,
                 ):
        super(SegmentationHead, self).__init__()
        self.classifier = nn.LazyLinear(n_classes, bias=True)

    def forward(self, x, mask=None):
        return self.classifier(x)