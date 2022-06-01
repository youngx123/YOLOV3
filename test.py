# -*- coding: utf-8 -*-
# @Author : Young
# @Time : 17:28  2021-08-26
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from model.YoLo import YOLOV3
from model.yolo_loss2 import YOLOVLoss as YOLOLoss2
from dataloader.yolo_dataloader import YOLODatasets
import imageio
from torch.utils.data import DataLoader
import cv2
import logging

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s %(message)s")


class Detection():
    def __init__(self, config, device="cpu"):
        self.CLASS_NAME = config.ClassNames
        self.CLASS_NUMBER = config.class_num
        self.conf_thres = config.confthred
        self.IMAGE_SIZE = config.train_size
        self.COLORS = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
                       range(self.CLASS_NUMBER)]
        self.net = YOLOV3(self.CLASS_NUMBER, pretrain=True, backbone=config.backbonName)
        pretrain_snapshot = config.pretrained_model
        # Restore weight
        if pretrain_snapshot:
            logging.info("Load pretrained weights from {}".format(pretrain_snapshot))
            state_dict = torch.load(pretrain_snapshot)
            self.net.load_state_dict(state_dict["model_weight"])
        else:
            logging.info("no trained weight find !")
            exit(0)

        self.device = torch.device(device)
        self.net = self.net.to(self.device)
        self.net.eval()

        # YOLO loss with 3 scales
        self.yolo_losses = [YOLOLoss2(config.Anchor[i], self.CLASS_NUMBER, self.IMAGE_SIZE) for i in range(3)]

        # DataLoader
        test_data = YOLODatasets(config.testFiles, self.IMAGE_SIZE, train=False)
        self.dataloader = torch.utils.data.DataLoader(test_data, batch_size=config.testBatchSize, shuffle=False,
                                                      num_workers=0, pin_memory=False)
        self.save_dir = config.savedir
        if not os.path.isdir(self.save_dir):  os.makedirs(self.save_dir)

    def Bbox_IOU(self, b1, b2):
        b1 = b1.reshape(-1, 7)
        xmin = torch.max(b1[:, 0], b2[:, 0])
        xmax = torch.min(b1[:, 2], b2[:, 2])
        ymin = torch.max(b1[:, 1], b2[:, 1])
        ymax = torch.min(b1[:, 3], b2[:, 3])

        wlenth = xmax - xmin
        hlenth = ymax - ymin

        union = wlenth * hlenth
        area1 = (b1[:, 3] - b1[:, 1]) * (b1[:, 2] - b1[:, 0])
        area2 = (b2[:, 3] - b2[:, 1]) * (b2[:, 2] - b2[:, 0])
        iou = union / (area1 + area2 - union)
        return iou

    def NMS(self, det_class, nms_thres):
        nms_result = []
        while (det_class.size(0)):
            nms_result.append(det_class[0].unsqueeze(0))
            if len(det_class) == 1:
                break
            det_class = det_class[1:]
            ious = self.Bbox_IOU(nms_result[-1], det_class)

            # Remove detections with IoU >= NMS threshold
            det_class = det_class[ious < nms_thres]
        return nms_result

    def Soft_NMS(self, det_class, nms_thres=0.001, sigma=0.5):
        nms_result = []
        score = det_class[..., 4]
        # # sort by score
        _, smax_orders = torch.sort(score, descending=True)
        smax_orders = smax_orders.data.cpu().numpy()
        while (smax_orders.size):
            i = smax_orders[0]
            nms_result.append(i)
            resut_index = np.delete(smax_orders, smax_orders == i)
            if not len(resut_index):
                break
            box1 = det_class[i]
            box2 = det_class[resut_index]

            ious = self.Bbox_IOU(box1, box2)
            weight = torch.exp(-(ious ** 2) / sigma)

            score[resut_index] = weight * score[resut_index]

            #
            thresh_mask = score[resut_index] > nms_thres
            thresh_mask = thresh_mask.data.cpu().numpy()
            smax_orders = resut_index[thresh_mask]

        return nms_result

    def Non_Max_Suppression(self, output, conf_thres=0.3, nms_thres=0.5):
        # # convert cxywh to (x1,y1, x2,y2)
        predict = output.new(output.shape)
        predict[:, :, 0] = output[:, :, 0] - output[:, :, 2] / 2
        predict[:, :, 1] = output[:, :, 1] - output[:, :, 3] / 2
        predict[:, :, 2] = output[:, :, 0] + output[:, :, 2] / 2
        predict[:, :, 3] = output[:, :, 1] + output[:, :, 3] / 2
        predict[:, :, 4:] = output[:, :, 4:]

        del output
        torch.cuda.empty_cache()

        result = [None for _ in range(len(predict))]
        for img_id, img_pred in enumerate(predict):
            predict_probClass = img_pred[..., 5:]
            class_conf, class_pred = torch.max(predict_probClass, 1, keepdim=True)
            obj_mask = ((img_pred[..., 4] * class_conf[:, 0]) > conf_thres).squeeze()

            img_pred = img_pred[obj_mask]
            pred_conf = class_conf[obj_mask].float()
            pred_class = class_pred[obj_mask].float()
            if not len(img_pred):
                continue

            detections = torch.cat((img_pred[:, :5], pred_conf, pred_class), 1)

            # # Soft-Nms
            # keep = self.Soft_NMS(detections)
            # result = detections[keep]

            # # NMS
            nms_result = self.NMS(detections, nms_thres)
            nms_result = torch.cat(nms_result).data

            # Add max detections to outputs
            result[img_id] = nms_result if result[img_id] is None else torch.cat((result[img_id], nms_result))
        return result

    def __call__(self):
        CLASS_NAME = self.CLASS_NAME
        import time
        startTime = time.time()
        with torch.no_grad():
            for step, samples in enumerate(self.dataloader):
                images, origin_size, filename = samples
                # print(filename)

                origin_size = (origin_size[0].item(), origin_size[1].item())
                images_data = images[0].numpy().transpose(1, 2, 0)
                images_data = cv2.resize(images_data, (origin_size))
                images_data22 = (255.0 * images_data).copy()
                images = images.float().to(self.device)
                outputs = self.net(images)
                output_list = [self.yolo_losses[i](outputs[i]) for i in range(3)]
                output = torch.cat(output_list, 1)

                batch_detections = self.Non_Max_Suppression(output, conf_thres=0.4, nms_thres=0.5)

                # write result images
                if batch_detections[0] is not None:
                    batch_detections = batch_detections[0].cpu().numpy().reshape(-1, 7)
                    for idx, detections in enumerate(batch_detections):
                        x1, y1, x2, y2, conf, cls_conf, cls_pred = detections
                        box_h = np.ceil(((y2 - y1) / self.IMAGE_SIZE) * origin_size[1])
                        box_w = np.ceil(((x2 - x1) / self.IMAGE_SIZE) * origin_size[0])
                        y1 = np.ceil((y1 / self.IMAGE_SIZE) * origin_size[1])
                        x1 = np.ceil((x1 / self.IMAGE_SIZE) * origin_size[0])

                        mess = "%s:%.2f" % (CLASS_NAME[int(cls_pred)], round(conf, 2))
                        cv2.rectangle(images_data22, (int(x1), int(y1)), (int(x1 + box_w), int(y1 + box_h)),
                                      self.COLORS[int(cls_pred)], 2)
                        cv2.putText(images_data22, mess, (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 255, 0), 1, 10)

                    imageio.imsave('{}/{}'.format(self.save_dir, os.path.basename(filename[0])),
                                   (images_data22).astype(np.uint8))

        endTime = time.time()
        print("run device :%s \ntotal % d  images use %.3f s, fps is : %.3f" % (
                            self.device, len(self.dataloader), endTime - startTime,
                            (len(self.dataloader) / (endTime - startTime))))


def main():
    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s %(message)s")

    import params
    config = params.parse_args()

    # Start test
    classDet = Detection(config)
    classDet()


if __name__ == "__main__":
    main()
