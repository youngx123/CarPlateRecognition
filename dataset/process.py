# -*- coding: utf-8 -*-
# @Author : Young
# @Time : 16:26  2022-05-20
import imageio
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
provinces2 = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫",
              "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
       'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def getSegImageMask(dir, file, saveImage, saveMaks):
    files = os.path.join(dir, file)
    data = imageio.imread(files)
    mask = np.zeros_like(data)

    file_split = file.split("-")
    points = file_split[3]
    label = file_split[4]

    points = points.split("_")
    pts = list()
    for pt in points:
        pt = pt.split("&")
        pts.append([pt[0], pt[1]])

    pts = np.array(pts, dtype=np.int)
    cv2.fillPoly(mask, [pts], (255, 255, 255))

    imageio.imsave(os.path.join(saveImage, file + ".png"), data)
    imageio.imsave(os.path.join(saveMaks, file + ".png"), mask)


def getCarNum(dir, file, saveImage):
    files = os.path.join(dir, file)
    data = imageio.imread(files)

    file_split = file.split("-")
    points = file_split[3]
    label = file_split[4]

    points = points.split("_")
    pts = list()
    for pt in points:
        pt = pt.split("&")
        pts.append([pt[0], pt[1]])

    pts = np.array(pts, dtype=np.int)

    r_b = pts[0]
    l_b = pts[1]
    l_u = pts[2]
    r_u = pts[3]
    platedata = data[l_u[1] - 3:r_b[1] + 3, l_u[0] - 3:r_b[0] + 3, :]

    width = int(math.sqrt((r_b[0] - l_b[0]) ** 2 + (r_b[1] - l_b[1]) ** 2) + 5)
    height = int(math.sqrt((l_b[0] - l_u[0]) ** 2 + (l_b[1] - l_u[1]) ** 2) + 5)

    # pts1 = np.float32([[l_u[1], l_u[0]], [r_u[1], r_u[0]], [l_b[1], l_b[0]], [r_b[1], r_b[0]]])
    pts1 = np.float32([[l_u[0], l_u[1]], [r_u[0], r_u[1]], [l_b[0], l_b[1]], [r_b[0], r_b[1]]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(data, M, (width, height))
    imgname = label
    imageio.imsave(os.path.join(saveImage, imgname + ".png"), warped)


def gen_dataset(dir, save):
    saveImage = os.path.join(save, "image")
    saveMaks = os.path.join(save, "mask")
    if not os.path.exists(saveImage) or not os.path.exists(saveMaks):
        os.makedirs(saveMaks, exist_ok=True)
        os.makedirs(saveImage, exist_ok=True)

    imageLists = os.listdir(dir)
    pl = Pool(processes=8)
    for file in tqdm(imageLists[:20000]):
        pl.apply_async(getCarNum, args=(dir, file, saveImage,))
        # pl.apply_async(getSegImageMask, args=(dir, file, saveImage,saveMaks,))

    pl.close()
    pl.join()


def CalculateShape(savedir):
    """
    Statistical average length and width
    :param savedir:
    :return:
    """
    imgfold = os.path.join(savedir, "image")
    imgLists = os.listdir(imgfold)
    h, w = 0, 0
    for file in imgLists:
        data = imageio.imread(os.path.join(imgfold, file))
        img_shape = data.shape[:2]
        h += img_shape[0]
        w += img_shape[1]

    print("mean shape : ", h / len(imgLists), w / len(imgLists))


def calculateNum(dirfold):
    dirfold = os.path.join(dirfold, "image1")
    imgList = os.listdir(dirfold )
    imgList = [os.path.join(dirfold, file) for file in imgList]
    pairs_num = dict()
    for file in imgList:
        basename = os.path.basename(file)
        basename = basename.split(".")[0].split("_")
        for index, id, in enumerate(basename):
            if index ==0:
                key = provinces2[int(id)]
            else:
                key = ads[int(id)]
            num = pairs_num.get(key, 0)
            # if key not in pairs_num:
            pairs_num[key] = num + 1

    y = pairs_num.values()
    x = pairs_num.keys()
    y = list(y)
    x = list(x)
    x0 = []
    y0=[]
    for id, xy in enumerate(zip(x,y)):
        xi, yi = xy
        x0.append(xi)
        y0.append(yi)
        if id %2==0:
            x0.append("")
            y0.append(0)
    plt.bar(x, y,lw=1)
    plt.xticks(rotation=30)
    plt.savefig("data_distribution.png")


if __name__ == '__main__':
    imgdir = r"E:\TSR\CCPD2020\ccpd_green\test"  # D:\MyNAS\CarPlate\dataset\CCPD2019.tar\CCPD2019\ccpd_fn D:\MyNAS\CarPlate\dataset\CCPD2019.tar\CCPD2019\ccpd_green\train
    savedir = r"E:\CarPlateRecognition\dataset\recgtrain"
    gen_dataset(imgdir, savedir)
    # calculateNum(savedir)
    # CalculateShape(savedir)
