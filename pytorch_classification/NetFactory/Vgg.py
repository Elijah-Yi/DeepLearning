#!/usr/local/bin/python3.9
# -*- coding:utf-8 -*-
"""
@Author   : Haiy. Yi
@Time     : 2024/1/29 11:03 PM
@File     : Vgg.py
@Software : PyCharm
@System   : MacOS catalina
"""
import sys

from NetFactory.utils import load_state_dict_from_url

sys.path.append('../')
sys.path.append('../../')
import torch.nn as nn
import torch
from torch.utils import model_zoo

from Utils.registries import MODEL_REGISTRY

__all__ = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
# official pretrain weights

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

import torchvision.models as models

models.vgg19()


def load_pretrained_weights(model, archme, load_fc=True):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    # AutoAugment or Advprop (different preprocessing)
    url_map_ = model_urls
    state_dict = model_zoo.load_url(url_map_[archme], map_location='cpu')
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        print("res.missing_keys:{}".format(res.missing_keys))
        assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
    print('Loaded pretrained weights for {}'.format(archme))


class VGG(nn.Module):
    def __init__(self, features, num_classes=None, cfg=None, init_weights=False):
        super(VGG, self).__init__()
        if cfg is not None:
            num_classes = num_classes if num_classes is not None else cfg['MODEL']['NUM_CLASSES']
        else:
            num_classes = 1000
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


@MODEL_REGISTRY.register()
def vgg11(cfg=None, archme="vgg11", pretrained: bool = True, progress: bool = True, **kwargs):
    assert archme in cfgs, "Warning: model number {} not in cfgs dict!".format(archme)
    arch = cfgs[archme]

    model = VGG(make_features(arch), cfg=cfg, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[archme],
                                              progress=progress)
        if cfg['MODEL']['NUM_CLASSES'] != 1000:
            state_dict.pop('classifier.6.weight')
            state_dict.pop('classifier.6.bias')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("missing_keys:{}".format(missing_keys))
        return model


@MODEL_REGISTRY.register()
def vgg13(cfg=None, archme="vgg13", pretrained: bool = True, progress: bool = True, **kwargs):
    assert archme in cfgs, "Warning: model number {} not in cfgs dict!".format(archme)
    arch = cfgs[archme]

    model = VGG(make_features(arch), cfg=cfg, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[archme],
                                              progress=progress)
        if cfg['MODEL']['NUM_CLASSES'] != 1000:
            state_dict.pop('classifier.6.weight')
            state_dict.pop('classifier.6.bias')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("missing_keys:{}".format(missing_keys))
        return model


@MODEL_REGISTRY.register()
def vgg16(cfg=None, archme="vgg16", pretrained: bool = True, progress: bool = True, **kwargs):
    assert archme in cfgs, "Warning: model number {} not in cfgs dict!".format(archme)
    arch = cfgs[archme]

    model = VGG(make_features(arch), cfg=cfg, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[archme],
                                              progress=progress)
        if cfg['MODEL']['NUM_CLASSES'] != 1000:
            state_dict.pop('classifier.6.weight')
            state_dict.pop('classifier.6.bias')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("missing_keys:{}".format(missing_keys))
    return model


@MODEL_REGISTRY.register()
def vgg19(cfg=None, archme="vgg19", pretrained: bool = True, progress: bool = True, **kwargs):
    assert archme in cfgs, "Warning: model number {} not in cfgs dict!".format(archme)
    arch = cfgs[archme]

    model = VGG(make_features(arch), cfg=cfg, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[archme],
                                              progress=progress)

        if cfg['MODEL']['NUM_CLASSES'] != 1000:
            state_dict.pop('classifier.6.weight')
            state_dict.pop('classifier.6.bias')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("missing_keys:{}".format(missing_keys))
    return model
