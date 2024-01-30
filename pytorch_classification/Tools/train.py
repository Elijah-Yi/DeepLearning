#!/usr/local/bin/python3.9
# -*- coding:utf-8 -*-
"""
@Author   : Haiy. Yi
@Time     : 2024/1/18 13:20
@File     : train.py
@Software : PyCharm
@System   : MacOS catalina
"""
import os
import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Utils.build_helper import build_model
from Tools.utils import load_config
from Utils.args_parse import parse_args
from Utils.distributed import init_distributed_mode
from Utils.build_helper import build_dataset, build_dataloader
from Utils.scheduler import build_scheduler

data_path = '../Dataset'


def main(args, cfg):
    init_distributed_mode(args, cfg)
    if cfg['DATASET']['DATASET_NAME'] == "CIFAR10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif cfg['DATASET']['DATASET_NAME'] == 'FLOWER':
        transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    else:
        transform = None
    train_set = build_dataset(mode='train', cfg=cfg, transform=transform)
    train_loader = build_dataloader(train_set, mode='train', cfg=cfg)

    val_set = build_dataset(mode='valid', cfg=cfg, transform=transform)
    val_loader = build_dataloader(val_set, mode='valid', cfg=cfg)

    model = build_model(cfg)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler, _ = build_scheduler(optimizer, cfg)
    best_acc = 0.0
    epochs = cfg['TRAIN']['MAX_EPOCH']
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        num_updates = epoch * len(train_loader)
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=180)

        for step, data in enumerate(train_bar):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step_update(num_updates=num_updates + step)
            scheduler.step(num_updates + step)

            predict_y = torch.max(outputs, dim=1)[1]
            accuracy = torch.eq(predict_y, labels).sum().item() / labels.size(0)

            # print statistics
            running_loss += loss.item()
            train_bar.desc = "Train epoch[{}/{}] loss:{:.3f} train_accuracy: {:.3f} local_rank:{}".format(epoch + 1, epochs, loss,
                                                                                                          accuracy, cfg['LOCAL_RANK'])
            # if step % 20 == 0:  # print every 500 mini-batches
            #     print('[%d, %5d] train_loss: %.3f  train_accuracy: %.3f' % (epoch + 1, step + 1, running_loss / 500, accuracy))
        model.eval()
        if epoch % cfg['TRAIN']['EVAL_PERIOD'] == 0:
            acc = 0.0  # accumulate accurate number / epoch
            nums = 0.0
            with torch.no_grad():
                val_bar = tqdm(val_loader, file=sys.stdout, ncols=180)
                for val_data in val_bar:
                    val_image, val_label = val_data
                    val_image = val_image.cuda(non_blocking=True)
                    val_label = val_label.cuda(non_blocking=True)
                    outputs = model(val_image)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_label).sum().item()
                    nums += val_label.size(0)
                    val_bar.desc = "Valid epoch[{}/{}] accuracy: {:.3f} local_rank:{}".format(epoch + 1, epochs, acc / nums,
                                                                                              cfg['LOCAL_RANK'])
            val_accurate = acc / nums
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / len(train_loader), val_accurate))
            if val_accurate > best_acc:
                best_acc = val_accurate
                os.makedirs(os.path.dirname(cfg['TRAIN']['CHECKPOINTS_SAVE_PATH']), exist_ok=True)
                torch.save(model.state_dict(), cfg['TRAIN']['CHECKPOINTS_SAVE_PATH'])

        # with torch.no_grad():
        #     for step, data in enumerate(val_loader, start=0):
        #
        #         outputs = model(val_image)  # [batch, 10]
        #         predict_y = torch.max(outputs, dim=1)[1]
        #         accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
        #
        #         print('[%d, %5d]  test_accuracy: %.3f' % (epoch + 1, step + 1, accuracy))
        #     running_loss = 0.0

    print('Finished Training')

    # os.makedirs(os.path.dirname(cfg['TRAIN']['CHECKPOINTS_SAVE_PATH']), exist_ok=True)
    # torch.save(model.state_dict(), cfg['TRAIN']['CHECKPOINTS_SAVE_PATH'])


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)
    from pprint import pformat

    print(pformat(cfg))
    main(args, cfg)
