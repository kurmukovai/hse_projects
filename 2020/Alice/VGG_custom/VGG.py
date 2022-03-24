import torch.nn as nn
import torch.nn.functional as F
import torchvision

def conv3x3(in_channels, out_channels, stride=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
        nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True)
    ]

class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            *conv3x3(1, 16),
            *conv3x3(16, 16),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            *conv3x3(16, 32),
            *conv3x3(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            *conv3x3(32, 64),
            *conv3x3(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            *conv3x3(64, 128),
            *conv3x3(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(5)

        self.classifier = nn.Sequential(
            nn.Linear(128 * 5 * 5, 1000),
            nn.ReLU(inplace = True),
            nn.Dropout(0.3),
            nn.Linear(1000, 128),
            nn.ReLU(inplace = True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x
