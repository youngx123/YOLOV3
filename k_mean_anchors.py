# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 16:55  2021-07-09
# @File Name: k_mean_anchors.py
import numpy as np
import xml.etree.ElementTree as ET
import glob
import random
import imageio

def iou(box, cluster):
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 - intersection)

    return iou


def avg_iou(box, cluster):
    return np.mean([np.max(iou(box[i], cluster)) for i in range(box.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(box, k):
    # 取出一共有多少框
    row = box.shape[0]

    # 每个框各个点的位置
    distance = np.empty((row, k))

    # 最后的聚类位置
    last_clu = np.zeros((row,))

    np.random.seed()

    # 随机选5个当聚类中心
    cluster = box[np.random.choice(row, k, replace=False)]
    # cluster = random.sample(row, k)
    while True:
        # 计算每一行距离五个点的iou情况。
        for i in range(row):
            distance[i] = 1 - iou(box[i], cluster)  # [sample_num, cluster_num]

        # 取出最小点
        near = np.argmin(distance, axis=1)  # [sample_num,], near值 0~cluster_num

        if (last_clu == near).all():  # [sample_num,]
            break

        # 求每一个类的中位点
        for j in range(k): # 选出near中值和0~k值 相同的类别
            cluster[j] = np.median(box[near == j], axis=0)

        last_clu = near

    return cluster


def load_xml_ann(path):
    data = []
    # 对于每一个xml都寻找box
    for xml_file in glob.glob('{}/*xml'.format(path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        # 对于每一个目标都获得它的宽高
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height
            # 得到宽高
            data.append([xmax - xmin, ymax - ymin])
    return np.array(data)


def load_txt_ann(path):
    with open(path, "r") as fid:
        data = fid.readlines()
    data = [b.strip().split(" ") for b in data]
    data_wh = []
    for boxs in data:
        img_path = boxs[0]
        image = imageio.imread(img_path)
        size = image.shape[:-1]
        boxs = [[float(bi) for bi in b.split(",")] for b in boxs[1:]]

        boxs = np.array(boxs)
        boxs = boxs.reshape(-1, 5)
        for b in boxs:
            xmin = np.float64(b[0]) /size[1]
            ymin = np.float64(b[1]) /size[0]
            xmax = np.float64(b[2]) /size[1]
            ymax = np.float64(b[3]) /size[0]

            data_wh.append([xmax - xmin, ymax - ymin])

    return np.array(data_wh)


if __name__ == '__main__':
    anchors_num = 9
    # 载入数据集，可以使用VOC的xml
    path = "voc2007train.txt"
    # path = "E:/Obj_D/yv3/2007_train.txt"
    # 载入所有的xml
    # 存储格式为转化为比例后的width,height
    # data = load_xml_ann(path)
    data = load_txt_ann(path)
    # 使用k聚类算法
    out = kmeans(data, anchors_num)
    out = out[np.argsort(out[:, 0])]
    print('acc:{:.2f}%'.format(avg_iou(data, out) * 100))
    print(out * np.array((640, 640)))
    print(out)