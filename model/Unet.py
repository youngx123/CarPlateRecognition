# -*- coding: utf-8 -*-
# @Author : Young
# @Time : 17:04  2022-05-24
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_channel, out_chanle, kernel_size=3, strid=1, padding=1):
        conv = nn.Conv2d(in_channel, out_chanle, kernel_size=kernel_size, stride=strid, padding=padding, bias=False)
        bn = nn.BatchNorm2d(out_chanle)
        super(ConvBnRelu, self).__init__(conv, bn)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.out_channels = [32, 32, 64, 128]
        self.stages1 = nn.Sequential(
            ConvBnRelu(3, self.out_channels[0], 3, 1, 1),
            ConvBnRelu(self.out_channels[0], self.out_channels[0], 3, 1, 1),
            ConvBnRelu(self.out_channels[0], self.out_channels[0], 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.stages2 = nn.Sequential(
            ConvBnRelu(self.out_channels[0], self.out_channels[1], 3, 1, 1),
            ConvBnRelu(self.out_channels[1], self.out_channels[1], 3, 1, 1),
            ConvBnRelu(self.out_channels[1], self.out_channels[1], 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.stages3 = nn.Sequential(
            ConvBnRelu(self.out_channels[1], self.out_channels[2], 3, 1, 1),
            ConvBnRelu(self.out_channels[2], self.out_channels[2], 3, 1, 1),
            ConvBnRelu(self.out_channels[2], self.out_channels[2], 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.stages4 = nn.Sequential(
            ConvBnRelu(self.out_channels[2], self.out_channels[3], 3, 1, 1),
            ConvBnRelu(self.out_channels[3], self.out_channels[3], 3, 1, 1),
            ConvBnRelu(self.out_channels[3], self.out_channels[3], 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x1 = self.stages1(x)
        x2 = self.stages2(x1)
        x3 = self.stages3(x2)
        x4 = self.stages4(x3)
        out = [x1, x2, x3, x4]
        return out


class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x_1 = nn.functional.max_pool2d(x, 3, stride=1, padding=1)
        x_2 = nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        x_3 = nn.functional.max_pool2d(x, 7, stride=1, padding=3)
        return torch.cat([x, x_1, x_2, x_3], dim=1)


class Decode(nn.Module):
    def __init__(self, chanle_list, num):
        super(Decode, self).__init__()

        self.spp = nn.Sequential(
            SPP(),
            ConvBnRelu(chanle_list[-1] * 4, chanle_list[-1], 3, 1, 1),
        )
        self.stages1 = nn.Sequential(
            ConvBnRelu(chanle_list[-1] + chanle_list[-2], chanle_list[-2], 1, 1, 0),
            ConvBnRelu(chanle_list[-2], chanle_list[-2], 3, 1, 1),
            ConvBnRelu(chanle_list[-2], chanle_list[-2], 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.stages2 = nn.Sequential(
            ConvBnRelu(chanle_list[-2] + chanle_list[-3], chanle_list[-3], 1, 1, 0),
            ConvBnRelu(chanle_list[-3], chanle_list[-3], 3, 1, 1),
            ConvBnRelu(chanle_list[-3], chanle_list[-3], 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.stages3 = nn.Sequential(
            ConvBnRelu(chanle_list[-3] + chanle_list[-4], chanle_list[-4], 1, 1, 0),
            ConvBnRelu(chanle_list[-4], chanle_list[-4], 3, 1, 1),
            ConvBnRelu(chanle_list[-4], chanle_list[-4], 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.stages4 = nn.Sequential(
            ConvBnRelu(chanle_list[-4], chanle_list[-4], 1, 1, 0),
            ConvBnRelu(chanle_list[-4], chanle_list[-4], 3, 1, 1),
            nn.Conv2d(chanle_list[-4], num, 1, 1, 0),
        )

    def forward(self, x):
        x0 = self.spp(x[-1])
        x0 = torch.cat((F.interpolate(x0, scale_factor=2), x[-2]), 1)
        x0 = self.stages1(x0)
        x0 = torch.cat((F.interpolate(x0, scale_factor=2), x[-3]), 1)
        x0 = self.stages2(x0)
        x0 = torch.cat((F.interpolate(x0, scale_factor=2), x[-4]), 1)
        x0 = self.stages3(x0)

        x0 = F.interpolate(x0, scale_factor=2)
        x0 = self.stages4(x0)
        return x0


class Unet(nn.Module):
    def __init__(self, class_num=1):
        super(Unet, self).__init__()

        self.encode = Encoder()
        self.encode_channels = self.encode.out_channels
        self.decode = Decode(self.encode_channels, class_num)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        # x = F.sigmoid(x)
        return x


if __name__ == '__main__':
    a = torch.randn((5, 3, 512, 512))
    net = Unet()
    out = net(a)
    torch.save(net, "a.pt")
