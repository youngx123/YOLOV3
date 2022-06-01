# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 18:04  2022-03-07

"""
yolov3 detection head
"""
from collections import OrderedDict
import torch
import torch.nn as nn


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


class FPN(nn.Module):
    def __init__(self,filterList):
        super(FPN, self).__init__()
        self.make_five_conv0 = make_five_conv([filterList[3], filterList[4]], filterList[4])

        self.upsample1 = Upsample(filterList[3], filterList[2])
        self.make_five_conv1 = make_five_conv([filterList[2], filterList[3]], filterList[3]+filterList[2])

        self.upsample2 = Upsample(filterList[2], filterList[1])
        self.make_five_conv2 = make_five_conv([filterList[1], filterList[2]], filterList[2]+filterList[1])

    def forward(self, x):
        x3, x4, P5 = x

        P5 = self.make_five_conv0(P5)
        # # FPN
        P5_upsample = self.upsample1(P5)
        P4 = torch.cat([x4, P5_upsample], axis=1)
        # 512 ->256 -> 512 -> 256 -> 512 -> 256
        P4 = self.make_five_conv1(P4)
        P3_upsample = self.upsample2(P4)

        P3 = torch.cat([x3, P3_upsample], axis=1)
        P3 = self.make_five_conv2(P3)

        return P3, P4, P5


class YOLOV3Head(nn.Module):
    def __init__(self, in_filters, out_filter):
        super(YOLOV3Head, self).__init__()
        self.fpn = FPN(in_filters)

        self.detc3 = nn.Sequential(
                            conv2d(in_filters[1], in_filters[2], 3),
                            nn.Conv2d(in_filters[2], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
                        )
        self.detc4 = nn.Sequential(
                            conv2d(in_filters[2], in_filters[3], 3),
                            nn.Conv2d(in_filters[3], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
                        )
        self.detc5 = nn.Sequential(
                            conv2d(in_filters[3], in_filters[4], 3),
                            nn.Conv2d(in_filters[4], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
                        )

    def forward(self, x):
        c5, c4, c3 = x
        c3, c4, c5 = self.fpn([c3, c4, c5])
        out5 = self.detc5(c5)
        out4 = self.detc4(c4)
        out3 = self.detc3(c3)
        return out5, out4, out3


if __name__ == '__main__':
    a1 = torch.randn((1, 128, 13, 13))
    a2 = torch.randn((1, 256, 26, 26))
    a3 = torch.randn((1, 512, 52, 52))

    net = YOLOV3Head([128,256,512], 75)
    out = net((a1, a2, a3))
    for a in out:
        print(a.shape)
