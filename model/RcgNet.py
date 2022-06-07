# -*- coding: utf-8 -*-
# @Author : Young
# @Time : 11:10  2022-05-13
import torch
import torch.nn as nn
# from .backbone import get_model
import torch.nn.functional as F
import math
from einops.layers.torch import Rearrange

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫",
             "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]

provinces2 = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫",
              "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
       'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x_1 = nn.functional.max_pool2d(x, 3, stride=1, padding=1)
        x_2 = nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        return torch.cat([x, x_1, x_2], dim=1)


class Conv2BnRelu(nn.Sequential):
    def __init__(self, in_chanle, outchannles):
        conv1 = nn.Conv2d(in_chanle, outchannles, kernel_size=3, stride=1, padding=1)
        conv2 = nn.Conv2d(outchannles, outchannles, kernel_size=3, stride=1, padding=1)
        bn = nn.BatchNorm2d(outchannles)
        relu = nn.ReLU()
        super(Conv2BnRelu, self).__init__(conv1, conv2, bn, relu)


class Conv3BnRelu(nn.Module):
    def __init__(self, in_chanle, outchannles):
        super(Conv3BnRelu, self).__init__()
        self.conv1 = nn.Conv2d(in_chanle, outchannles, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_chanle, outchannles, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.conv3 = nn.Conv2d(in_chanle, outchannles, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.bn = nn.BatchNorm2d(outchannles)
        self.relue = nn.ReLU()
        self.conv = nn.Conv2d(outchannles, outchannles, kernel_size=1, stride=1, padding=0)

        # self.resconv = nn.Conv2d(in_chanle, outchannles, kernel_size=1, stride=1, groups=in_chanle, padding=0)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2(x)
        x2 = self.conv3(x)
        x = x0 + x1 + x2
        x = self.relue(self.bn(self.conv(x)))
        return x


class MaxPoooling(nn.Sequential):
    def __init__(self, channel):
        avpool = nn.AvgPool2d(kernel_size=2, stride=2)
        conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        super(MaxPoooling, self).__init__(avpool, conv)


class BiLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BiLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class Recg(nn.Module):
    def __init__(self, image_size, class_num=len(provinces2) + len(ads), ctc=False):
        super(Recg, self).__init__()
        self.ctc = ctc
        self.image_size = image_size
        self.avg_size = (self.image_size[0] // 32, self.image_size[1] // 32)
        self.class_num = class_num
        self.filters = [16, 32, 32, 64, 128]
        self.layer1 = Conv2BnRelu(3, self.filters[0])
        self.pool1 = MaxPoooling(self.filters[0])

        self.layer2 = Conv3BnRelu(self.filters[0], self.filters[1])
        self.pool2 = MaxPoooling(self.filters[1])

        self.layer3 = Conv3BnRelu(self.filters[1], self.filters[2])
        self.pool3 = MaxPoooling(self.filters[2])

        self.layer4 = Conv3BnRelu(self.filters[2], self.filters[3])
        self.pool4 = MaxPoooling(self.filters[3])

        self.layer5 = Conv3BnRelu(self.filters[3], self.filters[4])
        self.pool5 = MaxPoooling(self.filters[4])
        if self.ctc:
            class_num +=1
            self.conv = nn.Sequential(
                Rearrange("b c h w-> b (h w) c"),
                Rearrange("b t c -> t b c"),
                BiLSTM(self.filters[4], self.filters[4], class_num),
            )
        else:
            self.conv = nn.Conv2d(self.filters[4], (class_num) * 9, kernel_size=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Module):
                for mi in m.modules():
                    if isinstance(mi, nn.Conv2d):
                        nn.init.xavier_normal(mi.weight.data)
                        mi.bias.data.fill_(0)
                    elif isinstance(mi, nn.Linear):
                        mi.weight.data.normal_()  # 全连接层参数初始化
                    elif isinstance(mi, nn.BatchNorm2d):
                        pass
            else:
                if isinstance(mi, nn.Conv2d):
                    # nn.init.normal(mi.weight.data)
                    nn.init.xavier_normal(mi.weight.data)
                    # nn.init.kaiming_normal(mi.weight.data)  # 卷积层参数初始化
                    mi.bias.data.fill_(0)
                elif isinstance(mi, nn.Linear):
                    mi.weight.data.normal_()  # 全连接层参数初始化

    def forward(self, feat: list):
        feat = self.pool1(self.layer1(feat))
        feat = self.pool2(self.layer2(feat))
        feat = self.pool3(self.layer3(feat))
        feat = self.pool4(self.layer4(feat))
        feat = self.pool5(self.layer5(feat))
        if self.ctc:
            feat = self.conv(feat)
            feat = F.log_softmax(feat, dim=2)
        else:
            feat = F.avg_pool2d(feat, kernel_size=self.avg_size, stride=self.avg_size)
            feat = self.conv(feat)
            feat = feat.view(-1, self.class_num, 9)
        return feat


if __name__ == '__main__':
    net = Recg((64, 192), ctc=True)
    a = torch.randn((8, 3, 64, 192))
    out = net(a)
    print(out.shape)
    torch.save(net, "reg.pt")
