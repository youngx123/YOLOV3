# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 15:41  2022-04-26

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init

"""
Mobile Net_V1:
Depthwise Convolution卷积大大减少运算量和参数数量

mobilenetV2:
Inverted Residuals(倒残差结构)
Linear Bottlenecks

mobilenetV3:
更新的block(bneck)，在倒残差结构上改动 -> 加入了SE模块，
更新了激活函数
使用NAS搜索参数重新设计耗时层结构

"""


# #https://github.com/xiaolai-sqlai/mobilenetv3/blob/master/mobilenetv3.py
class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_size, expand_size, out_size, kernel_size, stride, nolinear, semodule):
        """
        :param in_size: input channel
        :param expand_size: middle channel
        :param out_size: finale out put channel
        :param kernel_size: Conv kernel size
        :param nolinear: activate function
        :param semodule: SE module
        :param stride: Conv stride
        1x1 升维
        3x3 DW
        1x1 降维
        """
        super(Block, self).__init__()
        self.stride = stride
        # # se module
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        # # dw
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear

        # #
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = None
        if stride == 1 and in_size == out_size:  # # shortcut is used only when stride=1 and input channel == output channel
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        if self.se != None:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.shortcut else out
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=None):
        super(MobileNetV3_Large, self).__init__()
        self.numClass = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(16, 16, 16, 3, stride=1, nolinear=nn.ReLU(inplace=True), semodule=None),
            Block(16, 64, 24, 3, stride=2, nolinear=nn.ReLU(inplace=True), semodule=None),
            Block(24, 72, 24, 3, stride=1, nolinear=nn.ReLU(inplace=True), semodule=None),
            Block(24, 72, 40, 5, stride=2, nolinear=nn.ReLU(inplace=True), semodule=SeModule(72)),
            Block(40, 120, 40, 5, stride=1, nolinear=nn.ReLU(inplace=True), semodule=SeModule(120)),
            Block(40, 120, 40, 5, stride=1, nolinear=nn.ReLU(inplace=True), semodule=SeModule(120)),
            Block(40, 240, 80, 3, stride=2, nolinear=hswish(), semodule=None),
            Block(80, 200, 80, 3, stride=1, nolinear=hswish(), semodule=None),  # 7
            Block(80, 184, 80, 3, stride=1, nolinear=hswish(), semodule=None),
            Block(80, 184, 80, 3, stride=1, nolinear=hswish(), semodule=None),
            Block(80, 480, 112, 3, stride=1, nolinear=hswish(), semodule=SeModule(480)),
            Block(112, 672, 112, 3, stride=1, nolinear=hswish(), semodule=SeModule(672)),
            Block(112, 672, 160, 5, stride=1, nolinear=hswish(), semodule=SeModule(672)),
            Block(160, 672, 160, 5, stride=2, nolinear=hswish(), semodule=SeModule(672)),  # 13
            Block(160, 960, 160, 5, stride=1, nolinear=hswish(), semodule=SeModule(960)),
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        if self.numClass:
            self.linear3 = nn.Linear(960, 1280)
            self.bn3 = nn.BatchNorm1d(1280)
            self.hs3 = hswish()
            self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        outresult = []
        for id, layer in enumerate(self.bneck):
            out = layer(out)
            if id in [5, 12, 14]:
                outresult.append(out)

        # out = self.hs2(self.bn2(self.conv2(out)))
        # if self.numClass:
        #     out = F.avg_pool2d(out, 7)
        #     out = out.view(out.size(0), -1)
        #     out = self.hs3(self.bn3(self.linear3(out)))
        #     out = self.linear4(out)
        return out

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=None):
        super(MobileNetV3_Small, self).__init__()
        self.numClass = num_classes
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(16, 16, 16, 3, stride=2, nolinear=nn.ReLU(inplace=True), semodule=SeModule(16)),
            Block(16, 72, 24, 3, stride=2, nolinear=nn.ReLU(inplace=True), semodule=None),
            Block(24, 88, 24, 3, stride=1, nolinear=nn.ReLU(inplace=True), semodule=None),
            Block(24, 96, 40, 5, stride=2, nolinear=hswish(), semodule=SeModule(96)),
            Block(40, 240, 40, 5, stride=1, nolinear=hswish(), semodule=SeModule(240)),
            Block(40, 240, 40, 5, stride=1, nolinear=hswish(), semodule=SeModule(240)),
            Block(40, 120, 48, 5, stride=1, nolinear=hswish(), semodule=SeModule(120)),
            Block(48, 144, 48, 5, stride=1, nolinear=hswish(), semodule=SeModule(144)),
            Block(48, 288, 96, 5, stride=2, nolinear=hswish(), semodule=SeModule(288)),
            Block(96, 576, 96, 5, stride=1, nolinear=hswish(), semodule=SeModule(576)),
            Block(96, 576, 96, 5, stride=1, nolinear=hswish(), semodule=SeModule(576)),
        )
        self.layers_out_filters = [24, 48, 24, 48, 96]

        self.conv2 = nn.Conv2d(96, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.hs2 = hswish()
        if self.numClass:
            self.linear3 = nn.Linear(576, 1280)
            self.bn3 = nn.BatchNorm1d(1280)
            self.hs3 = hswish()
            self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        # out = self.bneck(out)
        outresult = []
        for id, layer in enumerate(self.bneck):
            out = layer(out)
            if id in [2, 7, 10]:
                outresult.append(out)

        # out = self.hs2(self.bn2(self.conv2(out)))
        return outresult

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = MobileNetV3_Small()
    net.eval()
    x = torch.randn(1, 3, 416, 416)
    y = net(x)

    # convert model to onnx format
    torch.onnx.export(net, x, "mobilev3.onnx", verbose=True,
                      training=torch.onnx.TrainingMode.TRAINING if False else torch.onnx.TrainingMode.EVAL,
                      input_names=["inputNode"], output_names=["outputNode"], opset_version=11)
    # print(y.size())
