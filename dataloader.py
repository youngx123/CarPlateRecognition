# -*- coding: utf-8 -*-
# @Author : Young
# @Time : 17:19  2022-05-20
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import imageio
import cv2
import albumentations as A

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫",
             "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]

provinces2 = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫",
              "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
       'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# # ctc loss data index
carNum = ["-"] + provinces2 + ads
provincesNum = len(provinces2)

class dataloder(Dataset):
    def __init__(self, dirpath, imagesize, number=None):
        super(dataloder, self).__init__()
        self.dirpath = dirpath
        self.imagesize = imagesize
        self.ImageList = os.listdir(self.dirpath)
        self.ImageList = [os.path.join(self.dirpath, path) for path in self.ImageList]
        self.MaskList = [path.replace("image", "mask") for path in self.ImageList]

        if number is None:
            self.num = len(self.ImageList)
        else:
            self.num = number
        self.ImageList = self.ImageList[:self.num]
        self.MaskList = self.MaskList[:self.num]
        self.transform = A.Compose([
            A.RandomScale(scale_limit=(0.9, 1.3)),
            A.RandomCrop(width=self.imagesize, height=self.imagesize),
            # A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=0.9),
            # A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.8),
            A.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        ])

    def __len__(self):
        return len(self.ImageList)

    def __getitem__(self, item):
        imgpath = self.ImageList[item]
        maskpath = self.MaskList[item]

        image = imageio.imread(imgpath)
        image = image / 255
        iH, iW = image.shape[:2]

        mask = imageio.imread(maskpath)
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mH, mW = mask.shape[:2]

        assert iW == mW and iH == mH, "image and mask shape error"

        scale_h = self.imagesize / iH
        scale_w = self.imagesize / iW
        scale = min(scale_h, scale_w)

        newH = self.imagesize  # int(scale * iH)
        newW = self.imagesize  # int(scale * iW)

        new_image = np.zeros((self.imagesize, self.imagesize, 3), dtype=np.float32)
        new_mask = np.zeros((self.imagesize, self.imagesize), dtype=np.float32)

        image = cv2.resize(image, (newW, newH))
        mask = cv2.resize(mask[..., 0], (newW, newH))

        new_image[:newH, :newW, :] = image[:, :, :]
        new_mask[:newH, :newW] = mask[:, :]
        if np.random.random() > 0.6:
            transformed = self.transform(image=new_image, mask=new_mask)
            new_image = transformed['image']
            new_mask = transformed['mask']

        new_image = new_image.transpose(2, 0, 1).astype(np.float)
        new_image = torch.from_numpy(new_image)

        new_mask = new_mask.astype(np.float)
        new_mask = new_mask[None, ...]
        new_mask = new_mask
        new_mask = torch.from_numpy(new_mask)

        return new_image, new_mask


class carNumloder(Dataset):
    def __init__(self, imagesize: tuple, dirpath=None, ImageList=None, number=None,ctc=None):
        super(carNumloder, self).__init__()
        self.dirpath = dirpath
        self.imagesize = imagesize
        self.ctc = ctc

        if ImageList is None:
            self.ImageList = os.listdir(self.dirpath)
            self.ImageList = [os.path.join(self.dirpath, path) for path in self.ImageList]
        else:
            self.ImageList = ImageList

        if number is None:
            self.num = len(self.ImageList)
        else:
            self.num = number

        self.ImageList = self.ImageList[:self.num]

        # self.transform = A.Compose([
        #     A.RandomScale(scale_limit=(0.9, 1.3)),
        #     A.RandomCrop(width=self.imagesize[1], height=self.imagesize[0]),
        #     A.OneOf([
        #         A.IAAAdditiveGaussianNoise(),
        #         A.GaussNoise(),
        #     ], p=0.2),
        #     A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=0.9),
        #     A.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        # ])

    def __len__(self):
        return len(self.ImageList)

    def __getitem__(self, item):
        imgpath = self.ImageList[item]
        basename = os.path.basename(imgpath)
        basename = basename.split(".")[0].split("_")

        labels = []
        for index, i in enumerate(basename):
            if index == 0:
                labels.append(float(i)+1)
            else:
                labels.append(float(i) + provincesNum +1)
        labels = np.array(labels)
        labels = labels.reshape(1, -1)
        if not self.ctc:
            if labels.shape[1] == 7:
                labels = np.concatenate((np.array(0).reshape(1, 1), labels, np.array(255).reshape(1, 1)), 1)
            if labels.shape[1] == 8:
                labels = np.concatenate((np.array(1).reshape(1, 1), labels), 1)
        else:
            if labels.shape[1] == 7:
                labels = np.concatenate((labels, np.array(255).reshape(1, 1)), 1)

        labels = labels[0]

        image = imageio.imread(imgpath)
        image = image / 255
        iH, iW = image.shape[:2]

        scale_h = self.imagesize[0] / iH
        scale_w = self.imagesize[1] / iW
        scale = min(scale_h, scale_w)

        newH = self.imagesize[0]  # int(scale * iH)
        newW = self.imagesize[1]  # int(scale * iW)

        new_image = np.zeros((self.imagesize[0], self.imagesize[1], 3), dtype=np.float32)

        image = cv2.resize(image, (newW, newH))

        new_image[:newH, :newW, :] = image[:, :, :3]
        # if np.random.random() > 0.6:
        #     transformed = self.transform(image=new_image)
        #     new_image = transformed['image']

        new_image = new_image.transpose(2, 0, 1).astype(np.float)
        new_image = torch.from_numpy(new_image)

        return new_image, torch.from_numpy(labels)


if __name__ == '__main__':
    # traindata = dataloder(imagesize=512, dirpath=r"D:\MyNAS\CarPlate\dataset\segtrain\image", number=None)
    traindata = carNumloder(imagesize=(64, 196), dirpath=r"E:\CarPlateRecognition\dataset\recgtrain\image", number=None, ctc=True)
    for i in range(1000):
        bathc = traindata.__getitem__(i)
