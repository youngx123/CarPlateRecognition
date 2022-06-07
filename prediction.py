# -*- coding: utf-8 -*-
# @Author : Young
# @Time : 15:16  2022-05-24
import torch
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os
from dataloader import provinces2, provincesNum, carNum


def evalSegModel(segpt):
    model = torch.load(segpt, map_location="cpu")
    model.eval()
    imagsize = 512

    imgpath = [
        "dataset/test/image/005-90_87-302&470_446&520-440&525_300&522_301&474_441&477-0_0_10_27_23_25_27-105-11.jpg.png",
        "dataset/test/image/0093-19_1-329&483_439&554-425&554_329&520_343&483_439&517-0_0_18_27_9_33_32-112-8.jpg.png",
        "dataset/test/image/0076-18_18-356&471_442&545-442&516_356&545_356&500_442&471-0_0_12_30_32_28_29-115-21.jpg.png",
        "dataset/test/image/01-90_265-231&522_405&574-405&571_235&574_231&523_403&522-0_0_3_1_28_29_30_30-134-56.jpg.png",
        "dataset/test/image/0056-18_11-304&528_375&595-369&595_304&573_310&528_375&550-0_0_21_28_26_27_15-91-26.jpg.png",
        "dataset/CCPD2019.tar/0936-5_33-70&396_568&553-568&553_132&513_70&396_506&436-0_0_3_26_26_29_15-41-49.jpg",
        "dataset/CCPD2019.tar/0936-7_3-118&420_550&601-542&601_118&546_126&420_550&475-0_0_2_11_27_29_33-57-25.jpg",
        "dataset/CCPD2019.tar/0944-2_7-142&421_593&596-579&575_142&596_156&442_593&421-0_0_24_17_33_28_33-190-105.jpg",
        "dataset/CCPD2019.tar/0947-18_16-179&406_523&636-518&636_179&520_184&406_523&522-0_0_7_15_30_26_32-64-54.jpg",
        "dataset/CCPD2019.tar/0946-0_0-124&447_621&606-620&606_124&602_125&447_621&451-0_0_27_28_30_27_27-113-25.jpg",
        "dataset/CCPD2019.tar/0944-15_34-240&501_625&706-589&608_240&706_276&599_625&501-0_0_2_31_24_24_27-61-67.jpg",

    ]
    for file in imgpath:
        data0 = imageio.imread(file)

        plt.subplot(1, 2, 1)
        plt.imshow(data0)

        iH, iW = data0.shape[:2]
        data = cv2.resize(data0, (imagsize, imagsize))
        data = data / 255.0
        data = data.transpose(2, 0, 1)
        data = torch.from_numpy(data).unsqueeze(0).float()

        scale_h = imagsize / iH
        scale_w = imagsize / iW
        scale = min(scale_h, scale_w)
        newH = imagsize  # int(scale * iH)
        newW = imagsize  # int(scale * iW)
        data = cv2.resize(data0, (newW, newH))
        new_image = np.zeros((imagsize, imagsize, 3))
        new_image[:newH, :newW, :] = data[:, :, :] / 255.0
        new_image = torch.from_numpy(new_image).permute(2, 0, 1).unsqueeze(0).float()
        pred = model(new_image)
        
        # # use softmax activation function 
        # pred = pred[0].permute(1, 2, 0).cpu().detach()
        # pred = F.softmax(pred, 1)
        # pred = torch.argmax(pred, 1)
        # pred = pred[0].detach().numpy() * 255
        # pred = np.array(pred).astype("uint8")
        
        # # use sigmoid activation function 
        pred = pred[0][0].detach().numpy()
        pred[pred >= 0.3] = 255
        pred[pred < 0.3] = 0
        pred2 = cv2.resize(pred, (iW, iH))
        pred2 = cv2.cvtColor(pred2, cv2.COLOR_GRAY2RGB)
        # weight = cv2.addWeighted(data0, 0.4, pred2, 0.6, 0)

        # pred = pred[0][0].cpu().detach()
        # pred[pred > 0.5] = 1
        plt.subplot(1, 2, 2)
        plt.imshow(pred2)
        plt.show()


def CTC_Decode(pred):
    _, preds = pred.max(2)
    pred = preds.transpose(1, 0).contiguous().view(-1)
    char_list = []
    for i in range(len(pred)):
        if pred[i] != 0 and (not (i > 0 and pred[i - 1] == pred[i])):
            char_list.append(pred[i].item())

    char_list1 = [i - 1 for i in char_list]
    char_list2 = [c - provincesNum for c in char_list1[1:]]
    result = [char_list1[0]] + char_list2
    return "_".join([str(i) for i in result])


def evalRcgModel(rcgpath):
    """
    1 represent green car
    :param rcgpath:
    :return:
    """

    model = torch.load(rcgpath, map_location="cpu")
    model.eval()
    imagsize = (64, 192)

    imgpath = [
        "dataset/test/rcgtest/0_0_0_0_24_30_26.png",
        "dataset/test/rcgtest/0_0_0_0_24_31_30.png",
        "dataset/test/rcgtest/0_0_0_0_25_32_27.png",
        "dataset/test/rcgtest/0_0_0_0_26_24_24.png",
        "dataset/test/rcgtest/0_0_0_0_27_28_27.png",
        "dataset/test/rcgtest/0_0_0_1_26_28_31.png",
        "dataset/test/rcgtest/29_0_5_25_31_29_29_29.png"
    ]
    imgdir = r"D:\MyNAS\CarPlate\dataset\test\rcgtest"
    imgpath = os.listdir(imgdir)
    imgpath = [os.path.join(imgdir, file) for file in imgpath]
    for file in imgpath:
        data0 = imageio.imread(file)

        plt.subplot(1, 2, 1)
        plt.imshow(data0)

        iH, iW = data0.shape[:2]
        data = cv2.resize(data0, (imagsize[1], imagsize[0]))
        data = data / 255.0
        data = data.transpose(2, 0, 1)
        data = torch.from_numpy(data).unsqueeze(0).float()

        pred = model(data)
        # #use ctc loss
        predres = CTC_Decode(pred)
        
        # #use softmax Classifier
        # pred = pred[0].permute(1, 0).cpu().detach()
        # pred = F.softmax(pred, 1)
        # pred = torch.argmax(pred, 1)
        # pred = pred.numpy()
        # car_category = pred[0]
        # predres = []
        # for idex, i in enumerate(pred[1:]):
        #     if idex == 0:
        #         i = i - 1
        #     if idex > 0:
        #         i -= 32
        #     predres.append(i)
        # if car_category == 0:
        #     predres = predres[:-1]
        label = os.path.basename(file).split(".")[0]
        label_lenth = [i for i in label.split("_")]
        print("input label : ", label, "蓝牌" if len(label_lenth) == 7 else "绿牌", end="\n")
        print("pred  label : ", predres, "蓝牌" if len(label_lenth) == 7 else "绿牌")
        print("*" * 40)


if __name__ == '__main__':
    segpath = "sigmoidModel.pt"
    # evalSegModel(segpath)

    rcgpath = "recg_best.pt"
    evalRcgModel(rcgpath)
