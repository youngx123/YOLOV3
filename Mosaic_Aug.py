# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 17:21  2022-03-10
import imageio
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

class MosaicAug():
    def __init__(self, imageList, annList, imgsize, scale=(0.4, 0.6)):
        self.imageList = imageList
        self.annList = annList
        self.imgsize = imgsize
        self.scale = scale
        self.index = 0

    def show_box(self, img, bboxs):
        for box in bboxs:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        plt.imshow(img)
        plt.show()

    def __call__(self, index):
        index = [index] + random.choices(range(len(self.imageList)), k=3)
        out_img = np.zeros((self.imgsize * 3, self.imgsize * 3, 3))
        imgHeight, imgWidth = out_img.shape[:2]
        out_ann = np.empty((0, 5))
        random.seed(100)
        scalex = self.scale[0] + random.random() * (self.scale[1] - self.scale[0])
        scaley = self.scale[0] + random.random() * (self.scale[1] - self.scale[0])

        pointx = int(scalex * imgWidth)
        pointy = int(scaley * imgHeight)

        for i in range(4):
            idx = index[i]
            imgpath = self.imageList[idx]
            annpath = self.annList[idx]
            data = imageio.imread(imgpath)
            ann = np.loadtxt(annpath, delimiter=",").reshape(-1, 5)
            annlabel = ann.copy()
            h, w = data.shape[:2]
            annlabel[:, [0, 2]] = annlabel[:, [0, 2]] / w
            annlabel[:, [1, 3]] = annlabel[:, [1, 3]] / h

            if i == 0:
                data = cv2.resize(data, (pointx, pointy))
                annlabel[:, [0, 2]] = annlabel[:, [0, 2]] * pointx
                annlabel[:, [1, 3]] = annlabel[:, [1, 3]] * pointy
                out_img[:pointy, :pointx, :] = data
                out_ann = np.vstack((out_ann, annlabel))

            if i == 1:
                data = cv2.resize(data, ((imgWidth - pointx), pointy))
                annlabel[:, [0, 2]] = annlabel[:, [0, 2]] * (imgWidth - pointx) + pointx
                annlabel[:, [1, 3]] = annlabel[:, [1, 3]] * pointy
                out_img[:pointy, pointx:, :] = data
                out_ann = np.vstack((out_ann, annlabel))

            if i == 2:
                data = cv2.resize(data, (pointx, (imgHeight - pointy)))
                annlabel[:, [0, 2]] = annlabel[:, [0, 2]] * pointx
                annlabel[:, [1, 3]] = annlabel[:, [1, 3]] * (imgHeight - pointy) + pointy
                out_img[pointy:, :pointx, :] = data
                out_ann = np.vstack((out_ann, annlabel))

            if i == 3:
                data = cv2.resize(data, ((imgWidth - pointx), (imgHeight - pointy)))
                annlabel[:, [0, 2]] = annlabel[:, [0, 2]] * (imgWidth - pointx) + pointx
                annlabel[:, [1, 3]] = annlabel[:, [1, 3]] * (imgHeight - pointy) + pointy
                out_img[pointy:, pointx:, :] = data
                out_ann = np.vstack((out_ann, annlabel))

        # # image before resize to train size
        # self.show_box(out_img / 255, out_ann)

        # #resize to train size
        out_img = cv2.resize(out_img, (self.imgsize, self.imgsize))
        scale_x = self.imgsize / imgWidth
        scale_y = self.imgsize / imgHeight

        out_ann[:, [0, 2]] = scale_x * out_ann[:, [0, 2]] / self.imgsize
        out_ann[:, [1, 3]] = scale_y * out_ann[:, [1, 3]] / self.imgsize

        # self.show_box(out_img / 255, out_ann)
        return out_img, out_ann



if __name__ == '__main__':
    print(random.random())
    pathlist = "voc2007val.txt"
    imgsize = 448
    scale = (0.3, 0.6)
    mosaic_aug = MosaicAug(pathlist, imgsize, scale)
    # a ,b = mosaic_aug(1)
    # a1,b1 = mosaic_aug(8)
    for i in range(20):
        print(i)
        # if i==3:
        mosaic_aug(i)
# mosaic_aug()
    # mosaic_aug()
    # mosaic_aug()

