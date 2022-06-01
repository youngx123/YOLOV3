# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 17:18  2021-07-07
import logging
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from .backbone.darknet import darknet53
from .yolov3_head import YOLOV3Head
from .backbone.mobilenet import MobileNetV3_Large, MobileNetV3_Small
from .backbone.ShuffleNetV2 import ShuffleNetV2


def conv2d(c_in, c_out, kernel):
    '''
    cbl = conv + batch_norm + leaky_relu
    '''
    pad = (kernel - 1) // 2 if kernel else 0
    conv1 = nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=1, padding=pad, bias=False)
    b1 = nn.BatchNorm2d(c_out)
    relu1 = nn.LeakyReLU(0.1)
    return nn.Sequential(OrderedDict([
        ("conv", conv1),
        ("bn", b1),
        ("relu", relu1),
    ]))


def make_pred_layer(in_filters, filters_list, out_filter):
    layers = nn.ModuleList([
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    ])
    return layers


class YOLOV3(nn.Module):
    def __init__(self, class_num, pretrain=None, backbone="darknet53"):
        super(YOLOV3, self).__init__()
        if backbone == "darknet53":
            print("use darknet53 as backbone")
            self.backbone = darknet53(pretrain)
        elif backbone == "MobileNetV3_Large":
            print("use MobileNetV3 Large as backbone")
            self.backbone = MobileNetV3_Large()
        elif backbone == "ShuffleNetV2":
            print("use ShuffleNetV2 Large as backbone")
            self.backbone = ShuffleNetV2()
        else:
            pass
        self.out_filters = self.backbone.layers_out_filters
        out_filter = 3 * (1 + 4 + class_num)

        # yolov3 head
        self.head = YOLOV3Head(self.out_filters, out_filter)

        # # # fpn + pan head
        # self.head = FPNPANHead(self.out_filters[2:][::-1], out_filter)

        # # yolox_head with 3 prediction 20, in_channels=[128, 256, 512]
        # self.head = YOLOXHead(class_num, in_channels=self.out_filters[2:][::-1])

    def forward(self, x):
        # # [batch_size, {1024,512, 256}, fsize,fsize ], fsize =[13, 26, 52]
        out3, out4, out5 = self.backbone(x)

        # # [batch_size, 75, fsize,fsize], fsize =[13, 26, 52]
        out5, out4, out3 = self.head((out5, out4, out3))

        return out5.float(), out4.float(), out3.float()


if __name__ == '__main__':
    A = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    anchor_grid = np.asarray(A, dtype=np.float32).reshape(3, 1, -1, 1, 1, 2)
    yolov3 = YOLOV3(2)
    model_dict = yolov3.state_dict()
    import numpy as np

    model_path = "./weights/darknet53_weights_pytorch.pth"
    print('Loading weights into state dict...')
    pretrained_dict = torch.load(model_path[0])
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}

    model_dict.update(pretrained_dict)
    yolov3.load_state_dict(model_dict)
    print('Finished!')

    from torchsummary import summary

    summary(yolov3, (3, 448, 448), device="cpu")
