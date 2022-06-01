# -*- coding: utf-8 -*-
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # # net params
    parser.add_argument('--backbonName', default="ShuffleNetV2", type=str,
                        choices=["darknet53", "ShuffleNetV2","MobileNetV3_Large", "MobileNetV3_Small"], help='backbone name')
    parser.add_argument('--class_num', default=20, type=int, help='dataset class number')
    parser.add_argument('--EPOCH', default=500, type=int, help='train epoch')
    parser.add_argument('--train_size', default=640, type=int, help='train epoch')
    parser.add_argument('--Anchor', default=[
                                            [[206, 154], [174, 298], [343, 330]],
                                            [[46, 133], [88, 93], [94, 207]],
                                            [[15, 27], [25, 72], [49, 43]]
                                        ],
                        help='anchors')

    parser.add_argument('--ClassNames', default=['aeroplane', 'bicycle', 'bird', 'boat',
                                                'bottle', 'bus', 'car', 'cat', 'chair',
                                                'cow', 'diningtable', 'dog', 'horse',
                                                'motorbike', 'person', 'pottedplant',
                                                'sheep', 'sofa', 'train', 'tvmonitor'
                                                ],
                        help='voc datasets class names')

    parser.add_argument('--device', default="cuda", type=str, choices=["cpu", "cuda"])
    parser.add_argument('--baseLr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--decayStep', default=100, type=int, help='initial weight decay step')

    # # dataset params
    parser.add_argument('--train_text', default="./2007train.txt", type=str, help='train list text')
    parser.add_argument('--val_text', default="./2007test.txt", type=str, help='validate list text')
    parser.add_argument('--batch_size', default=30, type=int, help='Batch size for training')
    parser.add_argument('--val_batch_size', default=6, type=int, help='Batch size for training')

    # # train params
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch to train')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--warm_up', default=5, type=int, help='lr warm up step')

    # # visualize params
    parser.add_argument('--eval_epoch', type=int, default=4, help='interval between evaluations')
    parser.add_argument('--save_folder', default='weights/', type=str, help='pt save dir')
    parser.add_argument('--show_loss_step', default=5, type=int, help='show training info')

    parser.add_argument('--pretrained_model', default=r"D:\MyNAS\SynologyDrive\yolov3\Imagsize_640_logs\model.pt", type=str, help='load pretrained model')

    # test params
    parser.add_argument('--testFiles', default="./2007test.txt", type=str, help='validate list text')
    parser.add_argument('--savedir', default="ShuffleNetV2_result", type=str,
                        help='save fold')
    parser.add_argument('--confthred', default=0.4, type=float)
    parser.add_argument('--NMSthred', default=0.4, type=float)
    parser.add_argument('--testBatchSize', default=1, type=int, help='Batch size for training')
    return parser.parse_args()

# # k-means on voc2007dataset anchors size
# "anchors": [[[206, 154], [174, 298], [343, 330]],
#             [[46, 133], [88, 93], [94, 207]],
#             [[15, 27], [25, 72], [49, 43]]],
# "anchors": [[[222, 432], [316, 240], [510, 508]],
#             [[78, 70], [117, 280], [144, 144]],
#             [[24, 40], [34, 90], [60, 174]]],

# k-means on voc2012 dataset anchors size
# "anchors": [[[209, 382], [320, 234], [424, 447]],
#             [[76, 68], [118, 283], [155, 147]],
#             [[25, 42], [37, 101], [67, 179]]],

# "anchors": [[[116, 90], [156, 198], [373, 326]],
#             [[30, 61], [62, 45], [59, 119]],
#             [[10, 13], [16, 30], [33, 23]]],
