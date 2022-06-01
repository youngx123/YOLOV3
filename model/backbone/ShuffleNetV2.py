# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 9:53  2022-05-23
import torch
import torch.nn as nn
from .ShuffleBlock import shuffleBlock


class ShuffleNetV2(nn.Module):
    def __init__(self, model_width=1):
        super(ShuffleNetV2, self).__init__()
        if model_width == 0.5:
            self.Output_channels = [3, 24, 48, 96, 192, 1024]
            self.layers_out_filters = self.Output_channels
        elif model_width == 1:
            self.Output_channels = [3, 24, 116, 232, 464, 1024]
            self.layers_out_filters = self.Output_channels
        elif model_width == 1.5:
            self.Output_channels = [3, 24, 176, 352, 704, 1024]
            self.layers_out_filters = self.Output_channels
        elif model_width == 2:
            self.Output_channels = [3, 24, 244, 488, 976, 2048]
            self.layers_out_filters =self.Output_channels

        self.repeat_num = [3, 3, 7]
        input_channel = self.Output_channels[1]

        self.convMaxpool = nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage2, input_channel = self.__MakeLayer(0, input_channel)
        self.stage2 = nn.Sequential(*self.stage2)

        self.stage3, input_channel = self.__MakeLayer(1, input_channel)
        self.stage3 = nn.Sequential(*self.stage3)

        self.stage4, input_channel = self.__MakeLayer(2, input_channel)
        self.stage4 = nn.Sequential(*self.stage4)

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, self.Output_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.Output_channels[-1]),
            nn.ReLU(inplace=True)
        )

    def __MakeLayer(self, i, input_channel):
        layer = []
        rptnum = self.repeat_num[i]
        output_channel = self.Output_channels[i + 2]
        layer.append(shuffleBlock(input_channel, output_channel, strid=2))
        input_channel = output_channel
        for _ in range(rptnum - 1):
            layer.append(shuffleBlock(input_channel // 2, output_channel, strid=1))
        return layer, output_channel

    def forward(self, x):
        x = self.convMaxpool(x)
        x2 = self.stage2(x)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        # x = self.conv_last(x)
        return [x2,x3,x4]


if __name__ == '__main__':
    net = ShuffleNetV2()
    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = net(test_data)
    # print(test_outputs.size())
