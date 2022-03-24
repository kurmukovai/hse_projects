import torch
import torch.nn as nn
from torch.nn import functional
import torchvision

def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class Interpolate(nn.Module):
    def __init__(self, size = None, scale_factor = None, mode = "nearest", 
                 align_corners = False):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels

        self.block = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear"),
            ConvRelu(in_channels, middle_channels),
            ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model = torchvision.models.vgg16_bn(pretrained=False)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=1, bias=True)
        
    def forward(self, x):
        x = self.model(x)
        return x
   
model = VGG16()
encoder = model.model.features

class UNet16(nn.Module):
    def __init__(self, num_classes = 1, num_filters = 64, pretrained = False):

        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = encoder
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder[0:5])

        self.conv2 = nn.Sequential(self.encoder[7:12])

        self.conv3 = nn.Sequential(self.encoder[14:22])

        self.conv4 = nn.Sequential(self.encoder[24:32])

        self.conv5 = nn.Sequential(self.encoder[34:42])

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, 
                                   num_filters * 8)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, 
                                   num_filters * 8)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, 
                                   num_filters * 2)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, 
                                   num_filters)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)
