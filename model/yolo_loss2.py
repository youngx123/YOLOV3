# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 11:59  2021-07-12
import math

import numpy as np
import torch
import torch.nn as nn


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


class YOLOVLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOVLoss, self).__init__()
        self.anchors = anchors
        self.MSELoss = nn.MSELoss(size_average=False)
        self.BCELoss = nn.BCELoss(size_average=False)

        self.num_classes = num_classes
        self.bbox_attrs = 1 + 4 + num_classes
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.img_size = img_size

        self.ignore_threshold = 0.5

    def forward(self, prediction, targets=None):
        BS = prediction.shape[0]
        anchors = self.anchors
        batch_size = prediction.size(0)
        grid_size = prediction.size(2)
        stride = self.img_size // grid_size
        num_anchors = len(anchors)

        prediction = prediction.view(batch_size, num_anchors, self.bbox_attrs,
                                     grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()

        scaled_anchors = torch.FloatTensor([(a[0] / stride, a[1] / stride) for a in anchors])

        # Sigmoid the  centre_X, centre_Y. and object confidencce
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        if targets is not None:
            targets = targets.cpu()
            mask, xywhconf, noobj_mask, tcls, box_loss_scale = self.buildTargets(targets=targets,
                                                                                 anchors=scaled_anchors,
                                                                                 num_anchors=num_anchors,
                                                                                 num_classes=self.num_classes,
                                                                                 grid_size=grid_size,
                                                                                 )

            noobj_mask = self.get_ignore(x, y, w, h, targets, scaled_anchors, grid_size, grid_size, noobj_mask)
            # # Handle masks
            xywhconf = xywhconf.to(x.device)
            tconf = xywhconf[..., -1].to(x.device)
            tcls = tcls.to(x.device)
            mask = xywhconf[..., 4] == 1
            objMask = mask.to(x.device)
            noobjMask = noobj_mask.to(x.device)
            # 权重
            box_loss_scale = 1.5 * (2 - box_loss_scale)
            box_loss_scale = box_loss_scale.type(FloatTensor)

            loss = 0
            # #计算 xywh 损失
            loss_x = self.BCELoss(x * objMask, xywhconf[..., 0] * objMask)  # * box_loss_scale[obj_mask]
            loss_y = self.BCELoss(y * objMask, xywhconf[..., 1] * objMask)
            loss_w = self.MSELoss(w * objMask, xywhconf[..., 2] * objMask)
            loss_h = self.MSELoss(h * objMask, xywhconf[..., 3] * objMask)
            loss_cord = self.lambda_xy * (loss_x + loss_y) + self.lambda_wh * (loss_h + loss_w)

            # # 置信度损失
            loss_obj = self.BCELoss(pred_conf * objMask, objMask * 1.0)
            loss_noObj = 0.5 * self.BCELoss(pred_conf * noobjMask, noobjMask * 0.0)
            loss_conf = loss_obj + loss_noObj

            loss_class = self.BCELoss(pred_cls[objMask == 1], tcls[objMask == 1])

            loss += loss_cord + loss_conf * self.lambda_conf + loss_class * self.lambda_cls

            return loss / BS

        else:
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

    def buildTargets(self, targets, anchors, num_anchors, num_classes, grid_size):
        batch_size = len(targets)

        mask = torch.zeros(batch_size, num_anchors, grid_size, grid_size, requires_grad=False)
        noobj_mask = torch.ones(batch_size, num_anchors, grid_size, grid_size, requires_grad=False)
        xywhconf = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 5, requires_grad=False)
        tcls = torch.zeros(batch_size, num_anchors, grid_size, grid_size, num_classes, requires_grad=False)
        # tconf = torch.zeros(batch_size, num_anchors, grid_size, grid_size, requires_grad=False)
        box_loss_scale = torch.zeros(batch_size, num_anchors, grid_size, grid_size, requires_grad=False)

        for batch_idx in range(batch_size):
            for target_idx in range(targets[batch_idx].shape[0]):
                if targets[batch_idx][target_idx].sum() == 0:
                    continue
                # convert to position relative to bounding box
                gx = targets[batch_idx][target_idx, 0] * grid_size
                gy = targets[batch_idx][target_idx, 1] * grid_size
                gw = targets[batch_idx][target_idx, 2] * grid_size
                gh = targets[batch_idx][target_idx, 3] * grid_size

                # get grid box indices
                gi, gj = int(gx), int(gy)

                '''
                get the anchor box that has the highest iou with [gw, gh]
                '''
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh], dtype=np.float)).unsqueeze(0)
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)),
                                                                  np.array(anchors)), 1))
                # get iou
                anchor_iou = bboxIOU_Patch(gt_box, anchor_shapes, True)
                anchor_iou = anchor_iou.reshape(-1, )
                noobj_mask[batch_idx, anchor_iou > self.ignore_threshold, gj, gi] = 0
                # best matching anchor box
                best = np.argmax(anchor_iou)
                '''
                计算偏移量
                '''
                xywhconf[batch_idx, best, gj, gi, 0] = gx - gi
                xywhconf[batch_idx, best, gj, gi, 1] = gy - gj
                xywhconf[batch_idx, best, gj, gi, 2] = math.log(gw / anchors[best][0] + 1e-16)
                xywhconf[batch_idx, best, gj, gi, 3] = math.log(gh / anchors[best][1] + 1e-16)
                xywhconf[batch_idx, best, gj, gi, 4] = 1
                # class num
                target_label = int(targets[batch_idx][target_idx, 4])
                # class label
                tcls[batch_idx, best, gj, gi, target_label] = 1
                # obj weights
                box_loss_scale[batch_idx, best, gj, gi] = targets[batch_idx][target_idx, 2] * targets[batch_idx][
                    target_idx, 3]
                # Masks
                mask[batch_idx, best, gj, gi] = 1

        return mask, xywhconf, noobj_mask, tcls, box_loss_scale

    def get_ignore(self, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        bs = len(targets)

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        # -----------------------------------------------------#
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(int(bs * len(scaled_anchors)), 1, 1) \
            .view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(int(bs * len(scaled_anchors)), 1, 1) \
            .view(y.shape).type(FloatTensor)

        #
        anchor_w = FloatTensor(scaled_anchors.cuda()).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors.cuda()).index_select(1, LongTensor([1]))

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        # -------------------------------------------------------#
        pred_boxes_x = torch.unsqueeze(x + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y + grid_y, -1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w) * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h) * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim=-1)

        for b in range(bs):
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)

            if len(targets[b]) > 0:
                gx = targets[b][:, 0] * in_w
                gy = targets[b][:, 1] * in_h
                gw = targets[b][:, 2] * in_w
                gh = targets[b][:, 3] * in_h
                gt_box = torch.cat([gx.reshape(-1, 1), gy.reshape(-1, 1), gw.reshape(-1, 1), gh.reshape(-1, 1)], -1)
                gt_box =gt_box.type(FloatTensor)

                gt_box = gt_box.to(pred_boxes_for_ignore.device)
                anch_ious = bboxIOU_Patch(gt_box, pred_boxes_for_ignore, True)
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0

        return noobj_mask


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

    return iou  # [aSize,bSize]
