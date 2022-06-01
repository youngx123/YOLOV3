# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 9:23  2022-02-26
import gc
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from model.utils import ModelEMA
import math
from model.YoLo import YOLOV3
from model.yolo_loss2 import YOLOVLoss as YOLOLoss2
from dataloader.yolo_dataloader import YOLODatasets
from model.Lossv3 import YOLOVLoss


def save_checkpoint(state_dict, optimizer, epoch, config, modelName):
    """
    state_dict : net state dict
    optimizer: optimizer parameters
    epoch : train epoch
    config : config files
    """
    checkpoint_path = os.path.join(config.weightFolds, f"{modelName}.pt")
    model_dict = {
        "model_weight": state_dict,
        "optimizer": optimizer.state_dict(),
        "step": epoch
    }
    torch.save(model_dict, checkpoint_path)
    return checkpoint_path


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_fit(config):
    # # get params
    device = config.device
    base_lr = config.baseLr
    classesNums = config.class_num
    pretrain_snapshot = config.pretrained_model
    decayStep = config.decayStep
    Anchors = config.Anchor
    trainSize = config.train_size
    EPOCH = config.EPOCH
    batchSize = config.batch_size
    valBatchSize = config.val_batch_size
    trainTextPath = config.train_text
    valTextPath = config.val_text
    showLossStep = config.show_loss_step

    net = YOLOV3(classesNums, True, backbone=config.backbonName)
    if pretrain_snapshot:
        logging.info("Load pretrained weights from {}".format(pretrain_snapshot))
        state_dict = torch.load(pretrain_snapshot)

        trainedDict = state_dict["model_weight"]
        netDict = net.state_dict()
        for k in trainedDict.keys():
            if k in netDict.keys() and trainedDict[k].shape == netDict[k].shape:
                netDict[k] = trainedDict[k]
        net.load_state_dict(netDict)
        logging.info("load pretrained model")

        del state_dict, trainedDict
        gc.collect()
        torch.cuda.empty_cache()

    optimizer = optim.SGD(net.parameters(), lr=base_lr)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decayStep)
    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss2(Anchors[i], classesNums, trainSize))

    # Datasets
    train_data = YOLODatasets(trainTextPath, trainSize)
    val_data = YOLODatasets(valTextPath, trainSize)

    dataloader = torch.utils.data.DataLoader(train_data, batch_size=batchSize, shuffle=True,
                                             num_workers=4, drop_last=True, pin_memory=False)
    valdataloader = torch.utils.data.DataLoader(val_data, batch_size=valBatchSize, shuffle=True,
                                                num_workers=4, drop_last=True, pin_memory=False)

    # # model start training
    net.to(device)
    net.train()
    net.float()
    best_val = np.inf
    nb = len(dataloader)  # number of batches
    n_burn = max(3 * nb, 500)
    lf = lambda x: (((1 + math.cos(x * math.pi / EPOCH)) / 2) ** 1.0) * 0.95 + 0.05  # cosine

    ema = ModelEMA(net)
    for epoch in range(EPOCH):
        for step, samples in enumerate(dataloader):
            images, labels = samples
            images = images.to(device)
            labels = labels.to(device)

            ni = step + nb * epoch  # number integrated batches (since train start)
            # warm-up
            if ni < n_burn * 2:
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, [0, n_burn], [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, [0, n_burn], [0.9, 0.937])
            if ni == n_burn:
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = base_lr

            # Forward and backward
            optimizer.zero_grad()
            outputs = net(images)

            loss = 0
            for i in range(3):
                _loss_item = yolo_losses[i](outputs[i], targets=labels)
                loss += _loss_item
            loss /= 3
            loss.backward()
            optimizer.step()

            if step % showLossStep == 0:
                _loss = loss.item()
                lr = optimizer.param_groups[0]['lr']
                logging.info("epoch [%.3d] iter = %d loss = %.2f lr = %.5f " % (epoch, step, _loss, lr))

        ema.update(net)
        if epoch % config.eval_epoch == 0:
            checkpointPath = save_checkpoint(net.state_dict(), optimizer, epoch, config, "model")
            logging.info("save model  checkpoint to %s" % checkpointPath)
            net.eval()
            val_loss = evaluate_mode(net, valdataloader, yolo_losses)
            if best_val > val_loss:
                logging.info("validate loss improve from  %.4f to  %.4f " % (best_val, val_loss))
                best_val = val_loss
                bestPath = save_checkpoint(net.state_dict(), optimizer, epoch, config, "best")
                logging.info("save best  checkpoint to %s" % bestPath)
            else:
                logging.warning("validate loss dont improve best :  %.4f , val loss : %.4f " % (best_val, val_loss))
            net.train()

        lr_scheduler.step()

    checkpointPath = save_checkpoint(net.state_dict(), optimizer, epoch, config, "last")
    logging.warning("last checkpoint saved to :  %s" % checkpointPath)


def evaluate_mode(model, valloader, yolo_losses, device="cuda"):
    model.eval()
    loss = 0
    with torch.no_grad():
        for step, samples in enumerate(valloader):
            images, labels = samples
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            for i in range(3):
                _loss_item = yolo_losses[i](outputs[i], targets=labels)
                loss += _loss_item
        loss = loss / (3*len(valloader))
    return loss.item()


def main():
    pil_logger = logging.getLogger("PIL")
    pil_logger.setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s %(message)s")

    import params
    config = params.parse_args()

    # Create weight_dir
    weight_dir = './Imagsize_{}_logs'.format(config.train_size)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    config.weightFolds = weight_dir
    logging.info("sub working dir: %s" % weight_dir)

    # Start training
    train_fit(config)


if __name__ == "__main__":
    main()
