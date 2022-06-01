# -*- coding: utf-8 -*-
# @Author : Young
# @Time : 15:35  2021-03-14
"""
.pth -> .onnx -> ( sim_onnx ) -> ncnn
"""

import os
import time
from torch.utils.data import DataLoader
import torch
from model.YoLo import YOLOV3
import numpy as np
import cv2
import onnx
from model.yolo_loss2 import YOLOVLoss as YOLOLoss2
from dataloader.yolo_dataloader import YOLODatasets
import ncnn
import imageio


class modelConvert():
    def __init__(self, weightPath):
        self.weightPath = weightPath
        device = "cuda"
        self.net = YOLOV3(config["yolo"]["classes"])

        # # load pratrained weight
        if config["pretrain_snapshot"]:
            print("Load pretrained weights from {}".format(config["pretrain_snapshot"]))
            state_dict = torch.load(config["pretrain_snapshot"])
            self.net.load_state_dict(state_dict["model_weight"])
        else:
            print("no trained weight find")
            exit(0)

        # YOLO loss with 3 scales
        self.yolo_losses = [YOLOLoss2(config["yolo"]["anchors"][i], config["yolo"]["classes"], config["img_w"]) for i in
                            range(3)]
        config["batch_size"] = 1

        # DataLoader
        test_data = YOLODatasets(config["test_path"], (config["img_w"], config["img_h"]), mode=False)
        self.dataloader = torch.utils.data.DataLoader(test_data, batch_size=config["batch_size"],
                                                      shuffle=False, num_workers=0, pin_memory=False)

    def convert2Onnx(self, onnxpath=None):
        dirname = os.path.dirname(onnxpath)
        if not os.path.exists(dirname): os.makedirs(dirname)

        out_name = "multi_class_classification.onnx"
        netInput = torch.randn(1, 3, 416, 416)
        inputName = ["Input_Image"]
        # outputName = ["out5", "out4", "out3"]
        outputName = ["out_pred"]
        self.net.eval()
        torch.onnx.export(self.net, netInput, onnxpath,
                          training=torch.onnx.TrainingMode.TRAINING if False else torch.onnx.TrainingMode.EVAL,
                          input_names=inputName, output_names=outputName, verbose=True, opset_version=11)

        print("convert to onnx")
        import onnxsim
        model_onnx = onnx.load(onnxpath)
        model_onnx, check = onnxsim.simplify(model_onnx)
        print("onnx simplify")
        assert check, 'assert check failed'
        onnx.save(model_onnx, onnxpath)

    def convert2Ncnn(self, onnxpath, ncnnPath):
        onnx2ncnn = r"D:\MyNAS\ncnn_windows_vs2017\x64\bin\onnx2ncnn.exe"
        outParamName = ncnnPath + ".param"
        outBinName = ncnnPath + ".bin"
        r_v = os.system(onnx2ncnn + " " + onnxpath + " " + outParamName + " " + outBinName)
        print(r_v)


class modelEval():
    def __init__(self, target_size=416):
        self.target_size = target_size
        valdir = r"D:\MyNAS\multi_class\data\test_data"
        # valset = ImageLoader(valdir)
        # self.valLoader = DataLoader(valset, batch_size=1, shuffle=False)

    def onnxEval(self, onnxpath):
        net = cv2.dnn.readNetFromONNX(onnxpath)
        if net:
            print("load model")

        with open("voc2007test.txt", "r") as fid:
            lines = fid.readlines()
        lines = [l.strip().split(" ")[0] for l in lines]
        for file in lines:
            data = imageio.imread(file)
            data = data / 255
            # data = data[..., np.newaxis]
            blob = cv2.dnn.blobFromImage(np.float32(data), 1, (self.target_size, self.target_size), (0, 0, 0, 0))
            # blob = blob.transpose(0,2,3,1)
            net.setInput(blob)
            # print(self.PB_Net.getUnconnectedOutLayersNames())
            pred = net.forward('out')
            pred = pred[0]

            max_idex = np.argmax(pred[..., 5:], 1)
            conf_obj = pred[4, :] * pred[5:, :][max_idex]
            for pred_item in pred:
                pass

    # def ncnnEval(self, ncnnParam, ncnnBin, saveDir):
    #     self.ncnnNet = ncnn.Net()
    #     self.ncnnNet.load_param(ncnnParam)
    #     self.ncnnNet.load_model(ncnnBin)
    #
    #     pred_num = 0
    #     startTime0 = time.time()
    #     for id, batch in enumerate(self.valLoader):
    #         startTime = time.time()
    #         _, label, imgfile = batch[0].float().to(), batch[1].float().to(), batch[2]
    #         label = label.squeeze(1)
    #         color_gt = label[0, :len(dataloder.COLOR)]
    #         type_gt = label[0, len(dataloder.COLOR):]
    #         _, color_gt_index = torch.max(color_gt, 0)
    #         _, type_gt_index = torch.max(type_gt, 0)
    #         imagename = "{}_{}_{}.png".format(dataloder.COLOR[int(color_gt_index)], dataloder.TYPE[int(type_gt_index)], str(id))
    #
    #         img0 = cv2.imread(imgfile[0])
    #         # img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    #         img = img0
    #         # img_h = img.shape[0]
    #         # img_w = img.shape[1]
    #
    #         mat_in = ncnn.Mat.from_pixels_resize(
    #             img,
    #             ncnn.Mat.PixelType.PIXEL_BGR,
    #             img.shape[1],
    #             img.shape[0],
    #             self.target_size,
    #             self.target_size,
    #         )
    #         mat_in.substract_mean_normalize([0, 0, 0], [1/255.0, 1/255.0, 1/255.0])
    #         # mat_in.substract_mean_normalize([255.0,255.0,255.0], [])
    #         ex = self.ncnnNet.create_extractor()
    #         ex.input("Input_Image", mat_in)
    #
    #         _, mat_out = ex.extract("Predict")
    #         pred = np.array(mat_out).reshape(1, -1)
    #         color_pred = pred[0, :len(dataloder.COLOR)]
    #         type_pred = pred[0, len(dataloder.COLOR):]
    #
    #         color_cls_index = np.argmax(color_pred, 0)
    #         color_max_prob = color_pred[color_cls_index]
    #
    #         type_cls_index = np.argmax(type_pred, 0)
    #         type_max_prob = type_pred[type_cls_index]
    #
    #         if color_cls_index == color_gt_index.item() and type_cls_index ==type_gt_index.item():
    #             pred_num += 1
    #         color_mess = '%s : %.3f' % (dataloder.COLOR[int(color_cls_index)], color_max_prob)
    #         cv2.putText(img0, color_mess, (int(10), int(100 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    #
    #         type_mess = '%s : %.3f' % (dataloder.TYPE[int(type_cls_index)], type_max_prob)
    #         cv2.putText(img0, type_mess, (int(10), int(200 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    #
    #         cv2.imwrite(saveDir + "/" + imagename, img0)
    #         endTime = time.time()
    #         print("%s use time %.3f"%(os.path.basename(imgfile[0]), endTime - startTime))
    #
    #     print("eval acc : %.3f , total num %d,  %d" % (pred_num / len(self.valLoader), pred_num, len(self.valLoader)))
    #     startTime2 = time.time()
    #     print(startTime2 - startTime0)


if __name__ == '__main__':
    import params

    config = params.params

    cvtModel = modelConvert(config)
    onnxsavepath = "./convetModel/yolov3.onnx"
    cvtModel.convert2Onnx(onnxsavepath)

    # ncnnsavepath = "./convetModel/yolov3_ncnn"
    # cvtModel.convert2Ncnn(onnxsavepath, ncnnsavepath)

    # ncnnParam = r"D:\MyNAS\multi_class\convert_model\multi_class_classification.param"
    # ncnnBin = r"D:\MyNAS\multi_class\convert_model\multi_class_classification.bin"
    # saveDir = "./test_result/"

    # onnxParam = "./convetModel/yolov3.onnx"
    # evalTools = modelEval()
    # evalTools.onnxEval(onnxParam)
    # evalTools.onnxEval()
    # evalTools.ncnnEval(ncnnParam, ncnnBin, saveDir)
