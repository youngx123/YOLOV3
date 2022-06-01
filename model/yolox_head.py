# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 17:59  2022-03-07

"""
use yolox detection head
Decoupled Head
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act, )
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[256, 512, 1024], act="silu", depthwise=False, ):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()  # 两个3x3的卷积

        self.cls_preds = nn.ModuleList()  # 一个1x1的卷积，把通道数变成类别数，比如coco 80类
        self.reg_preds = nn.ModuleList()  # 一个1x1的卷积，把通道数变成4通道，因为位置是xywh
        self.obj_preds = nn.ModuleList()  # 一个1x1的卷积，把通道数变成1通道，判断有无目标
        self.stems = nn.ModuleList()      # 模前面的 BaseConv模块

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width),
                         ksize=1, stride=1, act=act))

            self.cls_convs.append(
                nn.Sequential(
                    *[Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                      Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                      ])
            )
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=num_classes*3, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=4*3, kernel_size=1, stride=1, padding=0)
            )

            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=3, kernel_size=1, stride=1, padding=0) # 3
            )

    def forward(self, inputs):
        # ---------------------------------------------------#
        #   inputs输入
        #   P3_out  fsize1, fsize1, 256
        #   P4_out  fsize2, fsize2, 512
        #   P5_out  fsize3, fsize3, 1024
        # ---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            x = self.stems[k](x)
            # ---------------------------------------------------#
            cls_feat = self.cls_convs[k](x)
            # get predict class category result
            cls_output = self.cls_preds[k](cls_feat)

            # get xywh predict result
            reg_feat = self.reg_convs[k](x)
            reg_output = self.reg_preds[k](reg_feat)

            #  get if contain object probability
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


if __name__ == '__main__':
    a5 = torch.randn((1, 1024, 13, 13))
    a4 = torch.randn((1, 512, 26, 26))
    a3 = torch.randn((1, 256, 52, 52))

    net = YOLOXHead(20, in_channels=[1024, 512, 256])
    out = net((a5, a4, a3))
    for a in out:
        print(a.shape)
