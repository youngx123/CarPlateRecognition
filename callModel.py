# -*- coding: utf-8 -*-
# @Author : Young
# @Time : 9:00  2022-05-30

import torch
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os


class CarPlateReceg():
    def __init__(self, segmodel, recgmodel, device="cuda"):
        self.segmodel = torch.load(segmodel, map_location=device)
        self.recgmodel = torch.load(recgmodel, map_location=device)
        self.seg_size = (512, 512)
        self.recg_size = (64, 192)

    def forward(self, imagefile):
        pred = self.plateSeg(imagefile)
        carnum = self.numRecognition(pred)

        return carnum

    def plateSeg(self, file):
        data0 = imageio.imread(file)
        iH, iW = data0.shape[:2]
        data0 = data0 / 255.0
        data = cv2.resize(data0, self.seg_size)
        data = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0).float()

        pred = self.segmodel(data)

        pred = F.softmax(pred, 1)
        pred = torch.argmax(pred, 1)
        pred = pred[0].detach().numpy() * 255
        pred = np.array(pred).astype("uint8")
        pred2 = cv2.resize(pred, (iW, iH))

        carimg = self.getWarp(pred2, data0)

        # pred2 = cv2.cvtColor(pred2, cv2.COLOR_GRAY2RGB)
        # weight = cv2.addWeighted(data0, 0.4, pred2, 0.6, 0)

        return carimg

    def numRecognition(self, predseg):
        data = cv2.resize(predseg, (self.recg_size[1], self.recg_size[0]))
        data = data / 255.0
        data = data.transpose(2, 0, 1)
        data = torch.from_numpy(data).unsqueeze(0).float()

        pred = self.recgmodel(data)

        pred = pred[0].permute(1, 0).cpu().detach()
        pred = F.softmax(pred, 1)
        pred = torch.argmax(pred, 1)
        pred = pred.numpy()
        car_category = pred[0]
        predres = []
        for idex, i in enumerate(pred[1:]):
            if idex == 0:
                i = i - 1
            if idex > 0:
                i -= 32
            predres.append(i)
        if car_category == 0:
            predres = predres[:-1]

        return predres

    def getWarp(self, img, image):
        conr, H = cv2.findContours(img)
        miniRect = cv2.minAreaRect(conr)


if __name__ == '__main__':
    carNumRecog = CarPlateReceg("segmodel_best.pt", "recg_best.pt")
    testDir = ""
    imgList = os.listdir(testDir)
    for imgfile in imgList:
        num = carNumRecog.forward(imgfile)
