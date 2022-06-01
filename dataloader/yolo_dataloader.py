# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 9:24  2021-08-26
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.data import Dataset
import imageio
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
from dataloader.Mosaic_Aug import MosaicAug
from .augmentations import SSDAugmentation


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))


def Show_Bbox(image, box, size=1, center_format=False, ):
    if center_format:
        for b in box:
            pts1 = (int((b[0] - b[2]) * size), int((b[1] - b[3]) * size))
            pts2 = (int((b[0] + b[2]) * size), int((b[1] + b[3]) * size))
            cv2.rectangle(image, pts1, pts2, color=(255, 0, 0))
    else:
        for b in box:
            pts1 = (int(b[0] * size), int(b[1] * size))
            pts2 = (int(b[2] * size), int(b[3] * size))
            cv2.rectangle(image, pts1, pts2, color=(255, 0, 0))

    plt.imshow(image)
    plt.show()


class YOLODatasets(Dataset):
    def __init__(self, txtname, imageSie=416, classNum=None,
                 train=True, scale=32, MaxObj=60):
        super(YOLODatasets, self).__init__()

        self.featureSize = imageSie // scale
        self.classNum = classNum
        self.IMAGESIZE = imageSie
        self.train = train
        self.MaxObj = MaxObj
        self.ImgList, self.LabelList = self.readText(txtname)

        if self.classNum is None:
            self.classNum = len(VOC_CLASSES)

        if self.train:
            self.SSDAug = SSDAugmentation(self.IMAGESIZE, mean=(0, 0, 0), std=(1, 1, 1))
            self.mosaicaug = MosaicAug(self.ImgList, self.LabelList, self.IMAGESIZE)

    def readText(self, path):
        self.ImgList = list()
        self.LabelList = list()
        with open(path, "r") as fid:
            labelInfo = fid.readlines()

        labelInfo = [line.strip() for line in labelInfo]
        for id, line in enumerate(labelInfo):
            line = line.strip()
            labelPath = line.replace("JPEGImages", "Label").replace("jpg", "txt")
            self.ImgList.append(line)
            self.LabelList.append(labelPath)
        return self.ImgList, self.LabelList

    def __len__(self):
        return len(self.ImgList)

    def drawBBox(self, data, imgInfo):
        for i in range(len(imgInfo)):
            item = imgInfo[i]

            cv2.rectangle(data, (int(item[0]), int(item[1])),
                          (int(item[2]), int(item[3])),
                          (255, 0, 0), 2)
        plt.imshow(data)
        plt.show()

    def __len__(self):
        return len(self.ImgList)

    def __getitem__(self, item):
        if self.train:
            if random.random() > 2:
                img, gt = self.mosaicaug(item)
                # Show_Bbox(img/255.0, gt, self.IMAGESIZE)
                origin_H, origin_W = img.shape[:2]
            else:
                imgPath = self.ImgList[item]
                labelPath = self.LabelList[item]

                if os.path.exists(imgPath) and os.path.exists(labelPath):
                    img = imageio.imread(imgPath)
                    img = np.array(img, dtype=np.float)
                    origin_H, origin_W = img.shape[:2]
                    gt = np.loadtxt(labelPath, delimiter=",").reshape(-1, 5)
                    gt[..., [0, 2]] = gt[..., [0, 2]] / origin_W
                    gt[..., [1, 3]] = gt[..., [1, 3]] / origin_H
                else:
                    raise "path don't exist, check again"
                    exit(0)

            img, box, label = self.SSDAug(img, gt[..., :4], gt[..., -1])
            # Show_Bbox(img, box, self.IMAGESIZE)
            img = img.transpose(2, 0, 1)
            gt = np.concatenate((box, label.reshape(-1, 1)), 1)  # [x1,y1,x2,y2, label]

            gt[..., [0, 1]] = (box[..., [0, 1]] + box[..., [2, 3]])*0.5
            gt[..., [2, 3]] = box[..., [2, 3]] - box[..., [0, 1]]
            num = 5
            filledGt = np.zeros((self.MaxObj, num), np.float32)
            filledGt[range(len(gt))[:self.MaxObj]] = gt[:self.MaxObj]
            return torch.from_numpy(img), torch.from_numpy(filledGt)
        else:
            imgPath = self.ImgList[item]
            img = imageio.imread(imgPath)
            H,W = img.shape[:2]
            img = img / 255.0
            img = cv2.resize(img, (self.IMAGESIZE, self.IMAGESIZE))
            img = img.transpose(2, 0, 1)
            return torch.from_numpy(img), (W, H), imgPath


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes


if __name__ == '__main__':
    annotation_path = "../2007test.txt"
    train_dataset = YOLODatasets(annotation_path, 416, 20, train="train")
    for i in train_dataset:
        pass
