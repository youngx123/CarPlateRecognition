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
import math

class CarPlateReceg():
    def __init__(self, segmodel, recgmodel, device="cpu"):
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
        coner = cv2.goodFeaturesToTrack(img, 4, 0.1, 10)
        coner0 = coner.reshape(-1, 2)
        conr, H = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        miniRect = cv2.minAreaRect(conr[0])

        rectBox = cv2.boxPoints(miniRect)

        box = sorted(rectBox, key=lambda xypts:xypts[0])
        left_box = box[:2]
        rigt_box = box[2:]

        left_box = sorted(left_box, key=lambda lbox:lbox[1])
        lbup, lbdown= left_box
        lbup, lbdown = lbup.astype(np.int), lbdown.astype(np.int)
        rigt_box = sorted(rigt_box, key=lambda rbox: rbox[1])
        rbup, rbdown = rigt_box
        rbup, rbdown = rbup.astype(np.int), rbdown.astype(np.int)

        car_img = image[lbup[1]: rbdown[1], lbup[0]:rbup[0], :]
        width = int(math.sqrt((rbdown[0] - lbdown[0]) ** 2 + (rbdown[1] - lbdown[1]) ** 2) + 3)
        height = int(math.sqrt((rbdown[0] - rbup[0]) ** 2 + (rbdown[1] - rbup[1]) ** 2) + 3)

        pts1 = np.float32([[lbup[0], lbup[1]], [rbup[0], rbup[1]], [lbdown[0], lbdown[1]], [rbdown[0], rbdown[1]]])
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(image, M, (width, height))

        cv2.drawContours(image, [np.int0(rectBox)], -1, (255, 0, 0), 1)

    def rotate(self, img, angle, centerxy=None):
        # grab the dimensions of the image and calculate the center of the
        # image
        (h, w) = img.shape[:2]
        if centerxy is None:
            (cX, cY) = (w // 2, h // 2)
        else:
            (cX, cY) = centerxy
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        # cv2.imshow("Rotated by 45 Degrees", rotated)
        # rotate our image by -90 degrees around the image
        # M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        return rotated

if __name__ == '__main__':
    carNumRecog = CarPlateReceg("sigmoidModel.pt", "recg_best.pt")
    imgpath = [
        #"dataset/test/image/005-90_87-302&470_446&520-440&525_300&522_301&474_441&477-0_0_10_27_23_25_27-105-11.jpg.png",
        # "dataset/test/image/0093-19_1-329&483_439&554-425&554_329&520_343&483_439&517-0_0_18_27_9_33_32-112-8.jpg.png",
        # "dataset/test/image/0076-18_18-356&471_442&545-442&516_356&545_356&500_442&471-0_0_12_30_32_28_29-115-21.jpg.png",
        # "dataset/test/image/01-90_265-231&522_405&574-405&571_235&574_231&523_403&522-0_0_3_1_28_29_30_30-134-56.jpg.png",
        # "dataset/test/image/0056-18_11-304&528_375&595-369&595_304&573_310&528_375&550-0_0_21_28_26_27_15-91-26.jpg.png",
        "dataset/CCPD2019.tar/0936-5_33-70&396_568&553-568&553_132&513_70&396_506&436-0_0_3_26_26_29_15-41-49.jpg",
        "dataset/CCPD2019.tar/0936-7_3-118&420_550&601-542&601_118&546_126&420_550&475-0_0_2_11_27_29_33-57-25.jpg",
        "dataset/CCPD2019.tar/0944-2_7-142&421_593&596-579&575_142&596_156&442_593&421-0_0_24_17_33_28_33-190-105.jpg",
        "dataset/CCPD2019.tar/0947-18_16-179&406_523&636-518&636_179&520_184&406_523&522-0_0_7_15_30_26_32-64-54.jpg",
        "dataset/CCPD2019.tar/0946-0_0-124&447_621&606-620&606_124&602_125&447_621&451-0_0_27_28_30_27_27-113-25.jpg",
        "dataset/CCPD2019.tar/0944-15_34-240&501_625&706-589&608_240&706_276&599_625&501-0_0_2_31_24_24_27-61-67.jpg",

    ]
    # imgList = os.listdir(testDir)
    for imgfile in imgpath:
        num = carNumRecog.forward(imgfile)
