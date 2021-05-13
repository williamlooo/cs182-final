import torch
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn

#UNET: https://developers.arcgis.com/python/guide/how-unet-works/
#PYTORCH NETWORK PARTS: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
class DoubleConv(nn.Module):
    """2x of (convolution, [BN], ReLU)"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Net(nn.Module):
    def __init__(self, n_classes, im_height, im_width):
        super(Net, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(base_model.children()) 
        self.n_channels = 3
        self.n_classes = n_classes

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)
        #nn.Linear(in_features=1024, out_features=n_classes),
        #nn.Softmax(),
        #in=230400 if imagesize = 60
        #in=262144 if imagesize = 64
        self.classifier = nn.Sequential(    
            nn.Linear(in_features=262144, out_features=1024),
            nn.Linear(in_features=1024, out_features=n_classes),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


def convreludropout(in_channels, out_channels, kernel, padding):
    """
    convolution->relu->dropout
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
    )
    
#unet incorporating resnet weights
#UNET https://arxiv.org/pdf/1505.04597.pdf
#bilinear interpolation: https://arxiv.org/pdf/1805.09233.pdf
#pytorch parts from: https://github.com/usuyama/pytorch-unet/
class ResNetUNet(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())                
        
        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convreludropout(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)        
        self.layer1_1x1 = convreludropout(64, 64, 1, 0)       
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)        
        self.layer2_1x1 = convreludropout(128, 128, 1, 0)  
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)        
        self.layer3_1x1 = convreludropout(256, 256, 1, 0)  
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convreludropout(512, 512, 1, 0)  
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_up3 = convreludropout(256 + 512, 512, 3, 1)
        self.conv_up2 = convreludropout(128 + 512, 256, 3, 1)
        self.conv_up1 = convreludropout(64 + 256, 256, 3, 1)
        self.conv_up0 = convreludropout(64 + 256, 128, 3, 1)
        
        self.conv_original_size0 = convreludropout(3, 64, 3, 1)
        self.conv_original_size1 = convreludropout(64, 64, 3, 1)
        self.conv_original_size2 = convreludropout(64 + 128, 64, 3, 1)
        
        self.conv_last = OutConv(64, n_classes)

        self.classifier = nn.Linear(in_features=819200, out_features=n_classes)

        
    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        
        layer0 = self.layer0(input)            
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)        
        layer4 = self.layer4(layer3)
        
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
 
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)        
        x = self.conv_last(x)   
        x = torch.flatten(x, 1)
        #print("x output shape: {}".format(x.shape))
        out = self.classifier(x)
        #print(out.shape)
        return out