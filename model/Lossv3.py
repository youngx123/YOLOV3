# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 14:28  2022-03-28

import glob
import random
from collections import defaultdict
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def get_yolo_layers(model):
    bool_vec = [x['type'] == 'yolo' for x in model.module_defs]
    return [i for i, x in enumerate(bool_vec) if x]  # [82, 94, 106] for yolov3


def wh_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou


def compute_loss(p, targets):  # predictions, targets
    FT = torch.cuda.FloatTensor if p[0].is_cuda else torch.FloatTensor
    loss, lxy, lwh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0])
    txy, twh, tcls, tconf, indices = targets
    MSE = nn.MSELoss()
    CE = nn.CrossEntropyLoss()
    BCE = nn.BCEWithLogitsLoss()

    # Compute losses
    # gp = [x.numel() for x in tconf]  # grid points
    for i, pi0 in enumerate(p):  # layer i predictions, i
        b, a, gj, gi = indices[i]  # image, anchor, gridx, gridy

        # Compute losses
        k = 1  # nT / bs
        if len(b) > 0:
            pi = pi0[b, a, gj, gi]  # predictions closest to anchors
            lxy += k * MSE(torch.sigmoid(pi[..., 0:2]), txy[i])  # xy
            lwh += k * MSE(pi[..., 2:4], twh[i])  # wh
            lcls += (k / 4) * CE(pi[..., 5:], tcls[i])

        # pos_weight = FT([gp[i] / min(gp) * 4.])
        # BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        lconf += (k * 64) * BCE(pi0[..., 4], tconf[i])
    loss = lxy + lwh + lconf + lcls

    # Add to dictionary
    d = defaultdict(float)
    losses = [loss.item(), lxy.item(), lwh.item(), lconf.item(), lcls.item()]
    for name, x in zip(['total', 'xy', 'wh', 'conf', 'cls'], losses):
        d[name] = x

    return loss, d


class YOLOVLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOVLoss, self).__init__()
        self.anchors = anchors
        self.MSELoss = nn.MSELoss()
        self.BCELoss = nn.BCELoss()

        self.num_classes = num_classes
        self.bbox_attrs = 1 + 4 + num_classes
        self.lambda_xy = 0.05
        self.lambda_wh = 0.05
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.img_size = img_size

        self.ignore_threshold = 0.5

        self.hyp = {'k': 10.39,  # loss multiple
                    'xy': 0.1367,  # xy loss fraction
                    'wh': 0.01057,  # wh loss fraction
                    'cls': 0.01181,  # cls loss fraction
                    'conf': 0.8409,  # conf loss fraction
                    'iou_t': 0.1287,  # iou target-anchor training threshold
                    'lr0': 0.001028,  # initial learning rate
                    'lrf': -3.441,  # final learning rate = lr0 * (10 ** lrf)
                    'momentum': 0.9127,  # SGD momentum
                    'weight_decay': 0.0004841,  # optimizer weight decay
                    }

    def forward(self, prediction, targets=None):
        if targets is not None:
            # Define criteria
            MSE = nn.MSELoss()
            BC = nn.BCELoss()
            CE = nn.CrossEntropyLoss()
            BCE = nn.BCEWithLogitsLoss()

            txy, twh, tcls, tconf, indices = self.build_targets(prediction, targets=targets)
            ft = torch.cuda.FloatTensor if prediction[0].is_cuda else torch.Tensor
            lxy, lwh, lcls, lconf = ft([0]), ft([0]), ft([0]), ft([0])

            bs = prediction[0].shape[0]
            k = self.hyp['k'] * bs  # loss gain
            for i, pi0 in enumerate(prediction):
                grid_num = pi0.shape[-1]
                anchor_num = len(self.anchors)
                pi0 = pi0.view(bs, anchor_num, self.bbox_attrs, grid_num, grid_num)
                pi0 = pi0.permute(0, 1, 3, 4, 2)
                b, a, gj, gi = indices[i]  # image, anchor, gridx, gridy
                tconf = torch.zeros_like(pi0[..., 0])  # conf

                # Compute losses
                if len(b):  # number of targets
                    pi = pi0[b, a, gj, gi]  # predictions closest to anchors
                    tconf[b, a, gj, gi] = 1  # conf
                    # pi[..., 2:4] = torch.sigmoid(pi[..., 2:4])  # wh power loss (uncomment)

                    lxy += (k * self.hyp['xy']) * MSE(torch.sigmoid(pi[..., 0:2]), txy[i].to(pi.device))  # xy loss
                    lwh += (k * self.hyp['wh']) * MSE(pi[..., 2:4], twh[i].to(pi.device))  # wh yolo loss
                    lcls += (k * self.hyp['cls']) * CE((pi[..., 5:]), tcls[i].to(pi.device))  # class_conf loss

                # pos_weight = ft([gp[i] / min(gp) * 4.])
                # BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                lconf += (k * self.hyp['conf']) * BCE(pi0[..., 4], tconf)  # obj_conf loss
            loss = lxy + lwh + lconf + lcls

            return loss
        else:
            anchors = self.anchors
            batch_size = prediction.size(0)
            grid_size = prediction.size(2)
            stride = self.img_size // grid_size
            num_anchors = len(anchors)

            prediction = prediction.view(batch_size, num_anchors, self.bbox_attrs, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()

            scaled_anchors = torch.FloatTensor([(a[0] / stride, a[1] / stride) for a in anchors])

            # Sigmoid the  centre_X, centre_Y. and object confidencce
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])
            w = prediction[..., 2]
            h = prediction[..., 3]
            pred_conf = torch.sigmoid(prediction[..., 4])
            pred_cls = torch.sigmoid(prediction[..., 5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            scaled_anchors = scaled_anchors.type(FloatTensor)

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            grid_x = torch.linspace(0, grid_size - 1, grid_size).repeat(grid_size, 1).repeat(
                int(batch_size * len(self.anchors)), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, grid_size - 1, grid_size).repeat(grid_size, 1).t().repeat(
                int(batch_size * len(anchors)), 1, 1).view(y.shape).type(FloatTensor)
            # 生成先验框的宽高
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))

            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, grid_size * grid_size).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, grid_size * grid_size).view(h.shape)

            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h
            prediction = torch.cat((pred_boxes.view(batch_size, -1, 4) * stride, \
                                    pred_conf.view(batch_size, -1, 1), \
                                    pred_cls.view(batch_size, -1, self.num_classes)), -1)

            prediction = prediction.contiguous().view(batch_size, grid_size * grid_size * num_anchors, self.bbox_attrs)
            return prediction

    def build_targets(self, pred, targets):
        # targets = [image, x, y, w, h, class]
        batchsize = pred[0].shape[0]
        labels = []
        for id, t in enumerate(targets):
            num = len(t)
            index = torch.tensor(id).repeat(num, 1)
            t2 = torch.cat((index, t), dim=1)
            labels.append(t2)
        targets = torch.cat(labels, 0)
        del labels
        # anchors = closest_anchor(model, targets)  # [layer, anchor, i, j]
        txy, twh, tcls, tconf, indices = [], [], [], [], []
        for i, ps in enumerate(pred):
            grid = ps.shape[-1]
            scale = self.img_size / grid
            anchor_vec = self.anchors[i]
            anchor_vec = torch.tensor(np.array(anchor_vec) / scale)

            # iou of targets-anchors

            gwh = targets[:, 3:5] * grid
            iou = [wh_iou(x, gwh).reshape(1,-1) for x in anchor_vec]
            iou = torch.cat(iou, 0)
            if True:
                an = len(anchor_vec)
                nt = len(targets)
                a = torch.arange(an).view(-1, 1).repeat(1, nt).view(-1)
                targets = targets.repeat(an, 1)
            else:  # use best anchor only
                iou, a = iou.max(0)  # best iou and anchor

            # reject below threshold ious (OPTIONAL)
            reject = True
            if reject:
                j = iou.view(-1) > 0.12
                t, a = targets[j], a[j]

            # Indices
            b = t[:, 0].long().t()  # target image,
            # class
            c = t[:, -1].long().t()
            gxy = t[:, 1:3] * grid
            gwh = t[:, 3:5] * grid
            gi, gj = gxy.long().t()  # grid_i, grid_j
            indices.append((b, a, gj, gi))

            # XY coordinates
            txy.append((gxy - gxy.floor()).float())

            # Width and height
            twh.append(((gwh / anchor_vec[a])).float())  # yolo method
            # twh.append(torch.sqrt(gwh / anchor_vec[a]) / 2)  # power method

            # Class
            tcls.append(c)

            # Conf
            tci = torch.zeros((batchsize, 3, grid, grid))
            tci[b, a, gj, gi] = 1  # conf
            tconf.append(tci)

        return txy, twh, tcls, tconf, indices


def bboxIOU_Patch(box1, box2, cxywh):
    if cxywh:
        # convert center to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

        box1 = torch.zeros_like(box1)
        box2 = torch.zeros_like(box2)
        box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
        del b1_x1, b1_y1, b1_x2, b1_y2
        del b2_x1, b2_y1, b2_x2, b2_y2

    aSize = len(box1)  #
    bSize = len(box2)

    max_xy = torch.min(box1[:, 2:].unsqueeze(1).expand(aSize, bSize, 2),
                       box2[:, 2:].unsqueeze(0).expand(aSize, bSize, 2))
    min_xy = torch.max(box1[:, :2].unsqueeze(1).expand(aSize, bSize, 2),
                       box2[:, :2].unsqueeze(0).expand(aSize, bSize, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]

    area_a = ((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

    iou = inter / (area_a + area_b - inter)

    return iou  # # [aSize, bSize]


if __name__ == '__main__':
    pass
    # Build targets
    # pred = model(imgs.to(device))
    # target_list = build_targets(model, targets, pred)
    #
    # # Compute loss
    # loss, loss_dict = compute_loss(pred, target_list)
