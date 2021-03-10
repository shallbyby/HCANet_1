#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
resnet = torchvision.models.resnet.resnet34(pretrained = True)


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
        # self.conv = _DSConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

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

class _DepthwiseSeparaConvASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_DepthwiseSeparaConvASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, inplanes, kernel_size=kernel_size, groups=inplanes,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.atrous_conv2 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.bn2 = BatchNorm(planes)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.atrous_conv2(x)
        x = self.bn2(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        
        # self.conv = double_conv(384,384,)
        self.conv = _DSConvBnReLU(448,384,3,1,0,1)

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
    
        
class FCNRes34(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        resnet = torchvision.models.resnet.resnet34(pretrained = True)
        
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]  #64x256x256
        self.input_pool = list(resnet.children())[3]   #64x128x128
        self.layer1 = resnet.layer1   #64x128x128
        self.layer2 = resnet.layer2   #128x64x64
        self.layer3 = resnet.layer3   #256x32x32
        
        self.layer4 = resnet.layer4   #512x16x16
        
        # self.ASPP0 = _ASPP(64,16,rates=[1,6,12])
        self.ASPP1 = _ASPP(64,16,rates=[1,6,12])
        self.ASPP2 = _ASPP(128,32,rates=[1,6,12])
        self.ASPP3 = _ASPP(256,64,rates=[1,6,12])
        self.ASPP4 = _ASPP(512,128,rates=[1,6,12])
        self.ASPP_Decoder = _ASPP_Decoder(256,64,rates=[1,6,12,18,24])
        self.fc1 = nn.Conv2d(384,256,kernel_size=1,padding=0,stride=1)
        
        self.up1 = UPBlock(512, 256)
        self.sm1 = double_conv(512,256)
        
        self.up2 = UPBlock(256,128)
        self.sm2 = double_conv(256,128)
        
        self.up3 = UPBlock(128,64)
        self.sm3 = double_conv(128,64)
        
        self.up4 = UPBlock(64,64)
        self.sm4 = double_conv(128,64)
        
        self.out =nn.Conv2d(64,n_classes,kernel_size=1,stride=1)
        
    def forward(self,x):
        x1 = self.input_block(x)   #64x256x256
        
        # x1_= self.ASPP0(x1)
        
        
        x2 = self.input_pool(x1)   #64x128x28
        x3 = self.layer1(x2)     #64x128x128
        
        x3_ = self.ASPP1(x3)
        
        x4 = self.layer2(x3)     #128x64x64
        
        x4_ = self.ASPP2(x4)
        
        x5 = self.layer3(x4)     #256x32x32
        
        x5_ = self.ASPP3(x5)
        
        x6 = self.layer4(x5)     #512x16x16
        
        # x6_ = self.ASPP4(x6)
        # x6 = self.ASPP4(x6)
        
        x = self.up1(x6)         #512x32x32
        x = torch.cat([x,x5_],dim=1)  #1024x32x32
        # x = torch.cat([x, x5], dim=1)  # 1024x32x32

        x = self.sm1(x)

        x = self.up2(x)          #128x64x64
        x = torch.cat([x,x4_],dim=1)
        # x = torch.cat([x, x4], dim=1)
        x = self.ASPP_Decoder(x)
        x = self.fc1(x)
        x = self.sm2(x)

        x = self.up3(x)          #64x128x128
        x = torch.cat([x,x3_],dim=1)
        # x = torch.cat([x, x3], dim=1)

        x = self.sm3(x)

        x = self.up4(x)          #64x256x256
        x = torch.cat([x,x1],dim=1)
        
        x = self.sm4(x)
       
        x = self.out(x)
        
        x = F.interpolate(x,size=(512,512),mode='bilinear')
        
        return x
    
model = FCNRes34().cuda()
inp = torch.rand((2,3,512,512)).cuda()
out = model(inp)
print(out.shape)


# # In[ ]:
#
#
# try:
#     get_ipython().system('jupyter nbconvert --to python UNet34_ASPP_Decoder.ipynb')
# except:
#     pass