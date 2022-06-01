仅供自己学习YOLOV3目标检测方法, 损失函数以及anchor分配的代码实现
模型训练过程中加入学习率余弦衰减以及指数滑动平均方法。

以YOLOv3 为基础网络，在检测部分，分别使用yolov3head、Panhead以及YOLOXhead对模型进行测试，并使用opencv和ncnn进行部署


__`ncnnInference` 使用ncnn和opencv对模型进行推理__

ncnn 使用原始模型输出， 维度为 `[batch_size, grid_num, grdi_num, 3*(4+1+20)] , grid_num 的取值范围为[13,26,52]`

整体流程为：

ncnn_result -> opencv数组格式 -> 数组逐像素获取所有波段的值，进行判断过滤掉 `conf*obj_prob < conf_thred`的值
-> NMS -> drawImage 。

接口为 `ncnnDetection` 其中`Process`为转成opencv数据格式进行处理， `ncnnMatRead`直接对 `ncnn` 预测数据格式进行处理。

onnx 模型包括后处理阶段，直接对结果进行处理即可， 维度为 `pred_Num x 3*(4+1+class_num)`, 其中 `pred_Num =  3x pow(targetSize / 32, 2)x pow(targetSize / 16, 2)x pow(targetSize / 8, 2)`。

接口为`onnxDetection`。


测试轻量化网络作为骨干网络的检测效果
![](https://github.com/youngx123/YOLOV3/blob/master/ShuffleNetV2_result/000017.jpg?raw=true)
![](https://github.com/youngx123/YOLOV3/blob/master/ShuffleNetV2_result/000034.jpg?raw=true)
![](https://github.com/youngx123/YOLOV3/blob/master/ShuffleNetV2_result/000161.jpg?raw=true)
![](https://github.com/youngx123/YOLOV3/blob/master/ShuffleNetV2_result/000113.jpg?raw=true)


参考项目：
>https://github.com/bubbliiiing/yolo3-pytorch
>https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py


