# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 9:53  2022-05-23
import torch
import torch.nn as nn


class shuffleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strid=1):
        super(shuffleBlock, self).__init__()
        self.stride = strid
        mid_channel = out_channel // 2
        if self.stride == 1:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1, groups=mid_channel, bias=False),
                nn.BatchNorm2d(mid_channel),

                nn.Conv2d(mid_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(inplace=True),
            )

        if self.stride == 2:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(inplace=True),

                nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=2, padding=1, groups=mid_channel, bias=False),
                nn.BatchNorm2d(mid_channel),

                nn.Conv2d(mid_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channel),
                nn.ReLU(inplace=True),
            )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),

            nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(x)
            return torch.cat((x_proj, self.layer(x)), 1)
        if self.stride == 2:
            main_brach = self.layer(x)
            shortcut = self.shortcut(x)
            return torch.cat((main_brach, shortcut), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]
