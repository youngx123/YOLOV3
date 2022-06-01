# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 18:16  2021-07-07
import sys

sys.path.append("..")
from utils.utils import bbox_iou
import torch.nn as nn
import torch
import math
import numpy as np
import random
import cv2

COUNT = 0


def clip_by_tensor(t, t_min, t_max):
    t = t.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def MSELoss(pred, target):
    return (pred - target) ** 2


def BCELoss(pred, target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

        self.ignore_threshold = 0.5
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.obj_scale = 1  # 5
        self.noobj_scale = 1  # 1

        self.mse_loss = MSELoss
        self.bce_loss = BCELoss

    def forward(self, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        prediction = input.view(bs, self.num_anchors, self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        if targets is not None:

            #  build target
            mask, noobj_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale = self.get_target(targets, scaled_anchors,
                                                                                            in_w, in_h,
                                                                                            self.ignore_threshold)
            mask, noobj_mask = mask.bool().cuda(), noobj_mask.bool().cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()
            box_loss_scale = box_loss_scale.cuda()

            # # #
            loss_x = self.lambda_xy * (self.bce_loss(x[mask], tx[mask]) * box_loss_scale[mask]).mean()
            loss_y = self.lambda_xy * (self.bce_loss(y[mask], ty[mask]) * box_loss_scale[mask]).mean()
            loss_w = self.lambda_wh * (self.mse_loss(w[mask], tw[mask]) * box_loss_scale[mask]).mean()
            loss_h = self.lambda_wh * (self.mse_loss(h[mask], th[mask]) * box_loss_scale[mask]).mean()

            loss_conf = torch.mean(self.bce_loss(conf[mask], tconf[mask])) + \
                        1 * torch.mean(self.bce_loss(conf[noobj_mask], tconf[noobj_mask]))

            loss_cls = self.lambda_cls * torch.mean(self.bce_loss(pred_cls[mask], tcls[mask]))

            #  total loss = losses * weight
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            return loss
            # return loss, loss_x.item(), loss_y.item(), loss_w.item(), \
            #        loss_h.item(), loss_conf.item(), loss_cls.item()
        else:
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            # Calculate offsets for each grid
            grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
                bs * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
                bs * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

            # Calculate anchor w, h
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
            anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

            # Add offset and scale with anchors
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

            # Results
            _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
            output = torch.cat((pred_boxes.view(bs, -1, 4) * _scale, conf.view(bs, -1, 1),
                                pred_cls.view(bs, -1, self.num_classes)), -1)
            return output.data

    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):
        bs = len(target)
        obj_mask = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, self.num_anchors, in_h, in_w, requires_grad=False)

        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False)

        box_loss_scale = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)

        for b in range(bs):
            for t in range(target[b].shape[0]):
                if target[b][t].sum() == 0:
                    continue

                # Convert to position relative to box
                gx = target[b][t, 0] * in_w
                gy = target[b][t, 1] * in_h
                gw = target[b][t, 2] * in_w
                gh = target[b][t, 3] * in_h

                # Get grid box indices
                gi = int(gx)
                gj = int(gy)

                # Get shape of gt box
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)

                # Get shape of anchor box
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(anchors)), 1))
                # Calculate iou between gt and anchor shapes
                anch_ious = bbox_iou(gt_box, anchor_shapes)

                # Where the overlap is larger than threshold set mask to zero (ignore)
                noobj_mask[b, anch_ious > ignore_threshold, gj, gi] = 0

                # Find the best matching anchor box
                best_n = np.argmax(anch_ious)

                # Masks
                obj_mask[b, best_n, gj, gi] = 1
                noobj_mask[b, best_n,gj, gi] = 0

                # object
                tconf[b, best_n, gj, gi] = 1

                # One-hot encoding of label
                tcls[b, best_n, gj, gi, int(target[b][t, 4])] = 1

                # Coordinates
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj

                # Width and height
                tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)

                box_loss_scale[b, best_n, gj, gi] = (2 - (gw /in_w )* (gh/in_h))
                # box_loss_scale[b, best_n, gj, gi] = math.sqrt(2 - target[b][t, 2] * target[b][t, 3])

        return obj_mask, noobj_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale


def save_feature(feature_map, target, imgsize, COUNT):
    index = random.randint(0, len(feature_map) - 1)
    f_size = feature_map.shape[2:]
    if f_size[0] == 76:
        # pass
        fmap = feature_map[index]

        scale = imgsize[0] / f_size[0]
        # t = t*23

        # fmap = torch.nn.functional.upsample(fmap.unsqueeze(0), scale_factor=scale).detach().cpu().numpy()[0]
        t = target[index] * f_size[0]

        images_data = fmap.detach().cpu().numpy().transpose(1, 2, 0)[:, :, :3]
        COUNT += 1
        for b in t:
            cx, cy, w, h, conf = b
            x1 = cx - w / 2
            y1 = cy - h / 2

            x2 = cx + w / 2
            y2 = cy + h / 2
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.rectangle(cv2.UMat(images_data * 255), (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.imwrite('../output/{}.jpg'.format(COUNT), cv2.cvtColor(images_data, cv2.COLOR_RGB2BGR))
        pass
