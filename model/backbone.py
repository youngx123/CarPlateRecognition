# -*- coding: utf-8 -*-
# @Author : Young
# @Time : 11:11  2022-05-13
import torch
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter


def vgg16(pretrained=True):
    model = models.vgg16(pretrained=pretrained)
    model = model.features
    backbone = IntermediateLayerGetter(model, {"9": 0, "16": 1, "23": 2, "30": 3})
    out_channels = [128, 256, 512, 512]
    return backbone, 4, out_channels


def mobilenet_v3(pretrained=True):
    model = models.mobilenet_v3_large(pretrained=pretrained)
    model = model.features
    backbone = IntermediateLayerGetter(model, {"2": 0, "4": 1, "7": 2, "16": 3})
    out_channels = [24, 40, 80, 960]
    # backbone = IntermediateLayerGetter(model, {"0": 0, "1": 1, "4": 2, "12": 3})
    # out_channels = [16, 16, 40, 576]
    return backbone, 4, out_channels


def SqueezeNet():
    model = models.SqueezeNet()
    model = model.features
    backbone = IntermediateLayerGetter(model, {"2": 0, "6": 1, "12": 2})
    out_channels = [96, 256, 512]
    return backbone, 3, out_channels


def get_model(model_name):
    if model_name == "vgg":
        return vgg16()
    elif model_name == "mobilenet":
        return mobilenet_v3()
    else:
        return SqueezeNet()


if __name__ == '__main__':
    a = SqueezeNet()
    inputa = torch.randn((1, 3, 224, 224))
    out= a[0](inputa)
    for i in out:
        print(out[i].shape)
