# -*- coding: utf-8 -*-
# @Author : Young
# @Time : 14:28  2022-03-16
import os
from tqdm import tqdm
import numpy as np
import cv2

print(cv2.__version__)
import imageio

CLASS_NAME = [
              'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor'
            ]


class modelEval():
    """
    网络输出结果已经处理， 即： 预测偏移加上格网坐标， 预测长宽* 先验框的长宽；
    坐标表示相对于输入图像大小的坐标
    """
    def __init__(self, target_size=416):
        self.target_size = target_size

    def onnxEval(self, onnxpath, conf_thred=0.4, nms_thred=0.5):
        net = cv2.dnn.readNetFromONNX(onnxpath)
        if net:
            print("load model")

        with open("voc2007test.txt", "r") as fid:
            lines = fid.readlines()
        lines = [l.strip().split(" ")[0] for l in lines]
        for file in tqdm(lines):
            basename = os.path.basename(file)
            data0 = imageio.imread(file)
            # data = data0 / 255
            orig_H, orig_W = data0.shape[:2]

            blob = cv2.dnn.blobFromImage(np.float32(data0), 1 / 255, (self.target_size, self.target_size), (0, 0, 0))
            net.setInput(blob, "Input_Image")
            pred = net.forward('out_pred')
            pred = pred[0]
            class_pred = pred[..., 5:]
            score = pred[..., 4]

            prob_score = np.amax(class_pred, 1)  # # 对应类别的概率值
            conf_obj = score * prob_score

            obj_mask = conf_obj > conf_thred  # # 大于conf_thred 的检测结果
            pred = pred[obj_mask]
            prob_score = prob_score[obj_mask]  # # 大于conf_thred 的对应类别的概率值
            # #

            xywh = pred[:, :5]
            boxes = np.zeros_like(xywh)
            boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
            boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
            boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
            boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
            xywh[:, :4] = boxes[:, :4]  # # x1,y1,x2,y2
            class_prob = pred[:, 5:]  # #
            class_cat = np.argmax(class_prob, 1)  # # class cat 对应的类别索引
            detections = np.concatenate((xywh, prob_score.reshape(-1, 1), class_cat.reshape(-1, 1)), 1)

            sort_score_index = np.argsort(detections[:, 4])
            sort_score_index = sort_score_index[::-1]
            detections = detections[sort_score_index]

            result = self.NMS(detections, nms_thres=nms_thred)

            # write result images
            if len(result):
                for idx, detections in enumerate(result):
                    x1, y1, x2, y2, conf, cls_conf, cls_pred = detections
                    y1 = np.ceil((y1 / self.target_size) * orig_H)
                    x1 = np.ceil((x1 / self.target_size) * orig_W)

                    y2 = np.ceil((y2 / self.target_size) * orig_H)
                    x2 = np.ceil((x2 / self.target_size) * orig_W)

                    mess = "%s : %f" % (CLASS_NAME[int(cls_pred)], conf)
                    cv2.rectangle(data0, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(data0, mess, (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

                imageio.imsave('{}/{}'.format("./onnx_result", basename), (data0).astype(np.uint8))

    def NMS(self, det_class, nms_thres=0.4):
        nms_result = []
        while (det_class.shape[0]):
            nms_result.append(det_class[0])
            if len(det_class) == 1:
                break
            det_class = det_class[1:]
            ious = self.Bbox_IOU(nms_result[-1], det_class)

            # Remove detections with IoU >= NMS threshold
            det_class = det_class[ious < nms_thres]
        return nms_result

    def Bbox_IOU(self, b1, b2):
        b1 = b1.reshape(-1, 7)
        xmin = np.maximum(b1[:, 0], b2[:, 0])
        xmax = np.minimum(b1[:, 2], b2[:, 2])
        ymin = np.maximum(b1[:, 1], b2[:, 1])
        ymax = np.minimum(b1[:, 3], b2[:, 3])

        wlenth = xmax - xmin
        hlenth = ymax - ymin

        union = wlenth * hlenth
        area1 = (b1[:, 3] - b1[:, 1]) * (b1[:, 2] - b1[:, 0])
        area2 = (b2[:, 3] - b2[:, 1]) * (b2[:, 2] - b2[:, 0])
        iou = union / (area1 + area2 - union)
        return iou


class Eval():
    """
    直接输出网络结果，不经过任何预处理
    """

    def __init__(self, target_size=416):
        self.target_size = target_size
        self.scale = [32, 16, 8]
        self.anchor = np.array(
            [[[206, 154], [174, 298], [343, 330]],
             [[46, 133], [88, 93], [94, 207]],
             [[15, 27], [25, 72], [49, 43]]]
        )

    def Sigmoid(self, x):
        x = 1 / (1 + np.exp(-x))
        return x

    def onnxEval(self, onnxpath, conf_thred=0.4, nms_thred=0.5):
        net = cv2.dnn.readNetFromONNX(onnxpath)
        if net:
            print("load model")

        with open("voc2007test.txt", "r") as fid:
            lines = fid.readlines()
        lines = [l.strip().split(" ")[0] for l in lines]
        for file in tqdm(lines):
            basename = os.path.basename(file)
            data0 = imageio.imread(file)
            orig_H, orig_W = data0.shape[:2]

            blob = cv2.dnn.blobFromImage(np.float32(data0), 1 / 255, (self.target_size, self.target_size), (0, 0, 0))
            net.setInput(blob, "Input_Image")
            pred = net.forward(["out5", "out4", "out3"])  # # [1, 75, grid_num ,grid_num ] grid_num in [13,26,52]
            pred = [p.transpose(0, 2, 3, 1) for p in pred]

            # # process
            process = []
            for id, pred_item in enumerate(pred):
                pred_item = pred_item[0]

                grid_h, grid_w = pred_item.shape[:2]
                scale = self.target_size // grid_w
                for gy in range(grid_h):
                    for gx in range(grid_w):
                        for i in range(3):
                            x = pred_item[gy, gx, i * 25: (i + 1) * 25][0]
                            y = pred_item[gy, gx, i * 25: (i + 1) * 25][1]
                            w = pred_item[gy, gx, i * 25: (i + 1) * 25][2]
                            h = pred_item[gy, gx, i * 25: (i + 1) * 25][3]
                            conf = pred_item[gy, gx, i * 25: (i + 1) * 25][4]
                            class_prob = pred_item[gy, gx, i * 25: (i + 1) * 25][5:]

                            x = self.Sigmoid(x)
                            y = self.Sigmoid(y)
                            conf = self.Sigmoid(conf)
                            class_prob = self.Sigmoid(class_prob)
                            class_cat = np.argmax(class_prob, 0)  # # 类别值
                            prob_score = np.amax(class_prob, 0)  # #类别对应的概率
                            if conf * prob_score < conf_thred:
                                continue
                            x = x + gx
                            y = y + gy
                            w, h = self.makegrid(w, h, id, i, scale)
                            process.append([x * scale, y * scale, w * scale, h * scale, conf, prob_score, class_cat])

            process = np.array(process)
            if len(process) == 0:
                continue
            xywh = process[:, :4]
            boxes = np.zeros_like(xywh)
            boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
            boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
            boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
            boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

            xywh[:, :4] = boxes[:, :4]  # # x1,y1,x2,y2
            prob_score = process[:, 4]  # # obj_conf
            class_cat = process[:, 5:]  # # class_conf class

            detections = np.concatenate((xywh, prob_score.reshape(-1, 1), class_cat), 1)
            sort_index = np.argsort(prob_score)
            sort_index = sort_index[::-1]
            detections = detections[sort_index]
            result = self.NMS(detections, nms_thres=nms_thred)

            # write result images
            if len(result):
                for idx, detections in enumerate(result):
                    x1, y1, x2, y2, conf, cls_conf, cls_pred = detections
                    y1 = np.ceil((y1 / self.target_size) * orig_H)
                    x1 = np.ceil((x1 / self.target_size) * orig_W)

                    y2 = np.ceil((y2 / self.target_size) * orig_H)
                    x2 = np.ceil((x2 / self.target_size) * orig_W)

                    mess = "%s : %f" % (CLASS_NAME[int(cls_pred)], conf)
                    cv2.rectangle(data0, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(data0, mess, (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

                imageio.imsave('{}/{}'.format("./onnx_result", basename), (data0).astype(np.uint8))

    def makegrid(self, w, h, id, i, scale):
        anchor_i = self.anchor[id][i]
        anchor_i = anchor_i / scale
        w = np.exp(w) * anchor_i[0]
        h = np.exp(h) * anchor_i[1]
        return w, h

    def NMS(self, det_class, nms_thres=0.4):
        nms_result = []
        while (det_class.shape[0]):
            nms_result.append(det_class[0])
            if len(det_class) == 1:
                break
            det_class = det_class[1:]
            ious = self.Bbox_IOU(nms_result[-1], det_class)

            # Remove detections with IoU >= NMS threshold
            det_class = det_class[ious < nms_thres]
        return nms_result

    def Bbox_IOU(self, b1, b2):
        b1 = b1.reshape(-1, 7)
        xmin = np.maximum(b1[:, 0], b2[:, 0])
        xmax = np.minimum(b1[:, 2], b2[:, 2])
        ymin = np.maximum(b1[:, 1], b2[:, 1])
        ymax = np.minimum(b1[:, 3], b2[:, 3])

        wlenth = xmax - xmin
        hlenth = ymax - ymin

        union = wlenth * hlenth
        area1 = (b1[:, 3] - b1[:, 1]) * (b1[:, 2] - b1[:, 0])
        area2 = (b2[:, 3] - b2[:, 1]) * (b2[:, 2] - b2[:, 0])
        iou = union / (area1 + area2 - union)
        return iou


if __name__ == '__main__':
    onnxParam = "./convetModel/yolov3.onnx"
    evalTools = modelEval()
    evalTools.onnxEval(onnxParam)

    # onnxParam = "./convetModel/yolov3_3.onnx"
    # evalTools = Eval()
    # evalTools.onnxEval(onnxParam)
