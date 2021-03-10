#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
resnet = torchvision.models.resnet.resnet101(pretrained = True)


class UPBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsampling_method="bilinear"):
        super().__init__()
        
        if upsampling_method =="conv_transpose":
            self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2,stride=2)
        elif upsampling_method =="bilinear":
            self.upsample = nn.Sequential(nn.Upsample(mode='bilinear',scale_factor=2),
                                         nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=1))
    def forward(self,up_x):
        x = self.upsample(up_x)
        return x
    
class _ConvBnReLU(nn.Sequential):
    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch))

        if relu:
            self.add_module("relu", nn.ReLU())
            
class _DSConvBnReLU(nn.Sequential):
    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_DSConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, in_ch, kernel_size, stride, groups=in_ch,padding=padding, dilation=dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(in_ch))
        self.add_module("relu", nn.ReLU())
        self.add_module("conv",
                        nn.Conv2d(in_ch, out_ch, 1, bias=False))
        self.add_module("bn", nn.BatchNorm2d(out_ch))

        if relu:
            self.add_module("relu", nn.ReLU())

class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h    

class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _DSConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _DSConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
#         self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))
        
        # self.conv = double_conv(in_ch,in_ch,)
        self.conv = _DSConvBnReLU(in_ch, in_ch, 3, 1, 0, 1)
        
        
    def forward(self, x):
        t = torch.cat([stage(x) for stage in self.stages.children()], dim=1)
        t = self.conv(t)
        return t
class _ASPP_Decoder(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP_Decoder, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _DSConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _DSConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))
        
        # self.conv = double_conv(in_ch,in_ch,)
        self.conv = _DSConvBnReLU(1792, 1792, 3, 1, 0, 1)
        
        
    def forward(self, x):
        t = torch.cat([stage(x) for stage in self.stages.children()], dim=1)
        t = self.conv(t)
        return t
    
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
        
class FCNRes101(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        resnet = torchvision.models.resnet.resnet101(pretrained = True)
        
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]  #64x256x256
        self.input_pool = list(resnet.children())[3]   #64x128x128
        self.layer1 = resnet.layer1   #256x128x128
        self.layer2 = resnet.layer2   #512x64x64
        self.layer3 = resnet.layer3   #1024x32x32
        
        self.layer4 = resnet.layer4   #2048x16x16
        
        self.ASPP1 = _ASPP(256,64,rates=[1,6,12])
        self.ASPP2 = _ASPP(512,128,rates=[1,6,12])
        self.ASPP3 = _ASPP(1024,256,rates=[1,6,12])
        # self.ASPP4 = _ASPP(2048,512,rates=[1,6,12])
        self.ASPP_Decoder = _ASPP_Decoder(1024,256,rates=[1,6,12,18,24])
        self.fc1 = nn.Conv2d(1792,1024,kernel_size=1,padding=0,stride=1)
        
        self.up1 = UPBlock(2048, 1024)
        self.sm1 = double_conv(2048,1024)
        
        self.up2 = UPBlock(1024,512)
        self.sm2 = double_conv(1024,512)
        
        self.up3 = UPBlock(512,256)
        self.sm3 = double_conv(512,256)
        
        self.up4 = UPBlock(256,128)
        self.sm4 = double_conv(192,64)
        
        self.out =nn.Conv2d(64,n_classes,kernel_size=1,stride=1)
        
    def forward(self,x):
        x1 = self.input_block(x)   #64x256x256       
        
        x2 = self.input_pool(x1)   #64x128x28
        x3 = self.layer1(x2)     #256x128x128
        
        x3_ = self.ASPP1(x3)
        
        x4 = self.layer2(x3)     #512x64x64
        
        x4_ = self.ASPP2(x4)
        
        x5 = self.layer3(x4)     #1024x32x32
        
        x5_ = self.ASPP3(x5)
        
        x6 = self.layer4(x5)     #2048x16x16
        
        # x6_ = self.ASPP4(x6)
        
        x = self.up1(x6)         #1024x32x32
        x = torch.cat([x,x5_],dim=1)  #2048x32x32

        x = self.sm1(x)

        x = self.up2(x)          #512x64x64
        x = torch.cat([x,x4_],dim=1)
        x = self.ASPP_Decoder(x)
        x = self.fc1(x)
        
        x = self.sm2(x)

        x = self.up3(x)          #256x128x128
        x = torch.cat([x,x3_],dim=1)

        x = self.sm3(x)

        x = self.up4(x)          #64x256x256
        x = torch.cat([x,x1],dim=1)
        
        x = self.sm4(x)
       
        x = self.out(x)
        
        x = F.interpolate(x,size=(512,512),mode='bilinear')
        
        return x
    
model = FCNRes101().cuda()
inp = torch.rand((2,3,512,512)).cuda()
out = model(inp)
print(out.shape)
