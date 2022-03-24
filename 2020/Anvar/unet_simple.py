import torch.nn as nn
from torch.nn import functional

def conv_3x3(in_c, out_c):
    return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
    )

def conv(in_c, out_c):
    return nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

class Unet(nn.Module):
    """
    16       (+)      16->8->1
      32     (+)     32
        64   (+)   64
          64 -> 64
    """
    
    def __init__(self, ):
        super().__init__()
        
        self.max_pool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down_conv_1 = conv_3x3(1, 16)
        self.down_conv_2 = conv_3x3(16, 32)
        self.down_conv_3 = conv_3x3(32, 64)
        self.bottleneck_conv = conv_3x3(64, 64)
        
        self.upsample_1 = conv(64, 64)
        self.up_conv_1 = conv_3x3(64, 32)
        self.upsample_2 = conv(32, 32)
        self.up_conv_2 = conv_3x3(32, 16)
        self.upsample_3 = conv(16, 16)
        self.up_conv_3 = conv_3x3(16, 8)
        
        self.segm = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 1, kernel_size=1)
        )
        
        
    def forward(self, x):
        
        # down/contracting
        x1 = self.down_conv_1(x)
        x2 = self.max_pool2x2(x1)
        
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool2x2(x3)
        
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool2x2(x5)
        
        x = self.bottleneck_conv(x6)
        
        # up/expansive
        x = self.upsample_1(functional.interpolate(x, x5.shape[2:], mode='bilinear', align_corners=False))
        x = self.up_conv_1(x+x5)
        
        x = self.upsample_2(functional.interpolate(x, x3.shape[2:], mode='bilinear', align_corners=False))
        x = self.up_conv_2(x+x3)
        
        x = self.upsample_3(functional.interpolate(x, x1.shape[2:], mode='bilinear', align_corners=False))
        x = self.up_conv_3(x+x1)

        # segm
        x = self.segm(x)
        
        return x