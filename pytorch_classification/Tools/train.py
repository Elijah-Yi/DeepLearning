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

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    # train_set = torchvision.datasets.CIFAR10(root=data_path, train=True,
    #                                          download=True, transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=48,
    #                                            shuffle=True, num_workers=0)
    train_set = build_dataset(mode='train', cfg=cfg, transform=transform)
    train_loader = build_dataloader(train_set, mode='train', cfg=cfg)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    # val_set = torchvision.datasets.CIFAR10(root=data_path, train=False,
    #                                        download=False, transform=transform)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
    #                                          shuffle=False, num_workers=0)

    val_set = build_dataset(mode='valid', cfg=cfg, transform=transform)
    val_loader = build_dataloader(val_set, mode='valid', cfg=cfg)

    model = build_model(cfg)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler, _ = build_scheduler(optimizer, cfg)

    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        num_updates = epoch * len(train_loader)

        for step, data in enumerate(train_loader, start=0):
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
            if step % 20 == 0:  # print every 500 mini-batches
                print('[%d, %5d] train_loss: %.3f  train_accuracy: %.3f' % (epoch + 1, step + 1, running_loss / 500, accuracy))
        if epoch % 20 == 0 and epoch != 0:  # print every 500 mini-batches
            print("********************* start evaluation *************************")
            with torch.no_grad():
                for step, data in enumerate(val_loader, start=0):
                    val_image, val_label = data
                    val_image = val_image.cuda(non_blocking=True)
                    val_label = val_label.cuda(non_blocking=True)
                    outputs = model(val_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d]  test_accuracy: %.3f' % (epoch + 1, step + 1, accuracy))
                running_loss = 0.0

    print('Finished Training')

    save_path = './Model/lenet/Lenet.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)
    from pprint import pformat

    print(pformat(cfg))
    main(args, cfg)
