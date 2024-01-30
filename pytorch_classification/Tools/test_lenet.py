#!/usr/local/bin/python3.9
# -*- coding:utf-8 -*-
"""
@Author   : Haiy. Yi
@Time     : 2024/1/24 10:37 PM
@File     : test_lenet.py
@Software : PyCharm
@System   : MacOS catalina
"""
import sys
sys.path.append('../')
sys.path.append('../../')
import torch
import torchvision.transforms as transforms
from PIL import Image

from NetFactory.LeNet import LeNet
def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load("../Model/lenet/Lenet.pth"))

    im = Image.open('../images/for_cafar10/dog.jpeg')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
