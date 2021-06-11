import torch.nn as nn
import torch.nn.functional as F
import torchvision

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = torchvision.models.vgg16_bn(pretrained=False)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=1, bias=True)
        
    def forward(self, x):
        x = self.model(x)
        return x
