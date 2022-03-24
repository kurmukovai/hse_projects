import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MobileNetv2(nn.Module):
    def __init__(self):
        super(MobileNetv2, self).__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=False)
        self.model.features[0] = nn.Conv2d(1, 32, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=True)
        
    def forward(self, x):
        x = self.model(x)
        return x
