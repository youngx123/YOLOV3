# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 18:04  2022-03-07
"""
fpn+pan + yolov3_head
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch.nn as nn
import torch
import torch.functional as F


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, groups=1, bias=False, act=True):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class conv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel=1, stride=1):
        super(conv2d, self).__init__()
        pad = (kernel - 1) // 2 if kernel else 0
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=stride, padding=pad, bias=False)
        self.b1 = nn.BatchNorm2d(c_out)
        self.relu1 = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.relu1(self.b1(self.conv1(x)))
        return x


def Conv_Five_Layer(in_filters, filters_list):
    convlayer = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return convlayer


def Conv_Det_Layer(filters_list, out_filter):
    detlayer = nn.Sequential(
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return detlayer


class FPNPan(nn.Module):
    def __init__(self, in_channale: list):
        super(FPNPan, self).__init__()
        self.Upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.c5conv = conv2d(in_channale[0], in_channale[0], kernel=1)

        self.c4conv = conv2d(in_channale[0] + in_channale[1], in_channale[1], kernel=1)
        self.c4 = Conv_Five_Layer(in_channale[1], [in_channale[1]//2, in_channale[1]])

        self.c3conv = conv2d(in_channale[1]//2 + in_channale[2], in_channale[2], kernel=1)
        self.c3 = Conv_Five_Layer(in_channale[2], [in_channale[2]//2, in_channale[2]])

        # pan
        self.c3downsample = conv2d(in_channale[2]//2, in_channale[2]//2, stride=2)

        self.panc4 = Conv_Five_Layer(in_channale[2]//2 + in_channale[1]//2, [in_channale[1]//2, in_channale[1]])
        self.c4downsample = conv2d(in_channale[1]//2, in_channale[1]//2, stride=2)

        self.panc5 = Conv_Five_Layer(in_channale[0] + in_channale[1]//2, [in_channale[0]//2, in_channale[0]])

    def forward(self, x):
        c5, c4, c3 = x
        # FPN
        c5 = self.c5conv(c5)

        upc5 = self.Upsample(c5)
        c4 = self.c4conv(torch.cat((upc5, c4), dim=1))
        c4 = self.c4(c4)

        upc4 = self.Upsample(c4)
        c3 = self.c3conv(torch.cat((c3, upc4), dim=1))
        c3 = self.c3(c3)

        # PAN
        c3down = self.c3downsample(c3)
        c4 = torch.cat([c3down, c4], axis=1)
        c4 = self.panc4(c4)

        c4downsample = self.c4downsample(c4)   # conv 3x3, stride=2
        c5 = torch.cat([c4downsample, c5], axis=1)
        c5 = self.panc5(c5)
        return c5, c4, c3


class FPNPANHead(nn.Module):
    def __init__(self, inchanel, classNum):
        super(FPNPANHead, self).__init__()
        self.fpn = FPNPan(inchanel)
        self.det5 = Conv_Det_Layer([inchanel[0]//2, inchanel[0]], classNum)
        self.det4 = Conv_Det_Layer([inchanel[1]//2, inchanel[1]], classNum)
        self.det3 = Conv_Det_Layer([inchanel[2]//2, inchanel[2]], classNum)

    def forward(self, x):
        c5, c4, c3 = self.fpn(x)
        out5 = self.det5(c5)
        out4 = self.det4(c4)
        out3 = self.det3(c3)
        return out5, out4, out3


if __name__ == '__main__':
    a5 = torch.randn((1, 1024, 13, 13))
    a4 = torch.randn((1, 512, 26, 26))
    a3 = torch.randn((1, 256, 52, 52))

    # net = FPNPan([128, 256, 512]) 256, 512, 1024
    net = FPNPANHead([1024, 512, 256], 25)
    out = net((a5, a4, a3))
    for a in out:
        print(a.shape)