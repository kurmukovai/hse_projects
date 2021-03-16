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
        self.down_conv_4 = conv_3x3(64, 128)
        self.down_conv_5 = conv_3x3(128, 256)
        self.bottleneck_conv = conv_3x3(256, 256)
        
        self.upsample_1 = conv(256, 256)
        self.up_conv_1 = conv_3x3(256, 128)
        self.upsample_2 = conv(128, 128)
        self.up_conv_2 = conv_3x3(128, 64)
        self.upsample_3 = conv(64, 64)
        self.up_conv_3 = conv_3x3(64, 32)
        self.upsample_4 = conv(32, 32)
        self.up_conv_4 = conv_3x3(32, 16)
        self.upsample_5 = conv(16, 16)
        self.up_conv_5 = conv_3x3(16, 8)
        
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
        
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool2x2(x7)
        
        x9 = self.down_conv_5(x8)
        x10 = self.max_pool2x2(x9)
        
        x = self.bottleneck_conv(x10)
        
        # up/expansive
        x = self.upsample_1(functional.interpolate(x, x9.shape[2:], mode='bilinear', align_corners=False))
        x = self.up_conv_1(x+x9)
        
        x = self.upsample_2(functional.interpolate(x, x7.shape[2:], mode='bilinear', align_corners=False))
        x = self.up_conv_2(x+x7)
        
        x = self.upsample_3(functional.interpolate(x, x5.shape[2:], mode='bilinear', align_corners=False))
        x = self.up_conv_3(x+x5)
        
        x = self.upsample_4(functional.interpolate(x, x3.shape[2:], mode='bilinear', align_corners=False))
        x = self.up_conv_4(x+x3)
        
        x = self.upsample_5(functional.interpolate(x, x1.shape[2:], mode='bilinear', align_corners=False))
        x = self.up_conv_5(x+x1)

        # segm
        x = self.segm(x)
        
        return x