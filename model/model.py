# -*- coding: utf-8 -*-
# @Author : Young
# @Time : 11:10  2022-05-13
import torch
import torch.nn as nn
from .backbone import get_model
import torch.nn.functional as F
import math


class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x_1 = nn.functional.max_pool2d(x, 3, stride=1, padding=1)
        x_2 = nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        return torch.cat([x, x_1, x_2], dim=1)


class Decode(nn.Module):
    def __init__(self, num, channels: list, class_num: int):
        super(Decode, self).__init__()
        self.layer = []
        for i in range(1, num):
            in_filter = channels[i - 1]
            out_filter = channels[i]
            in_filter = in_filter + out_filter

            self.layer.append(
                nn.Sequential(*[
                    nn.Conv2d(in_filter, out_filter, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_filter),
                    nn.ReLU(),

                    nn.Conv2d(out_filter, out_filter, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_filter),
                    nn.ReLU()])
            )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.spp = SPP()
        self.conv1x1 = nn.Conv2d(channels[0] * 3, channels[0], 1)
        self.convout = nn.Conv2d(channels[-1], class_num, 1)

    def forward(self, feat: list):
        x = self.spp(feat[3])
        x = self.conv1x1(x)
        for id, layers in enumerate(self.layer):
            id = 2 - id
            x = self.upsample(x)
            x = torch.cat([x, feat[id]], dim=1)
            x = layers.to(x.device)(x)

        x = F.interpolate(x, scale_factor=4)
        x = self.convout(x)
        return x


class Unet(nn.Module):
    def __init__(self, image_size, class_num, backbone="mobilenet"):
        super(Unet, self).__init__()

        self.image_size = image_size
        self.backbone, self.featNum, self.outchannels = get_model(backbone)

        self.decode = Decode(self.featNum, self.outchannels[::-1], class_num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feat):
        feat = self.backbone(feat)
        out = self.decode(feat)
        out = F.sigmoid(out)
        return out


class EAST(nn.Module):
    def __init__(self, scope=512):
        super(EAST, self).__init__()

        self.conv1 = nn.Conv2d(32, 1, 1)
        # self.sigmoid1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(32, 4, 1)
        # self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(32, 1, 1)
        # self.sigmoid3 = nn.Sigmoid()
        self.scope = scope

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        score = self.conv1(x)
        loc = self.conv2(x)
        angle = self.conv3(x)
        geo = torch.cat((loc, angle), 1)
        return score, geo


if __name__ == '__main__':
    net = Unet(448, 2)
    a = torch.randn((1, 3, 448, 448))
    out = net(a)
    torch.save(net, "ad.pt")
