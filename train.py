# -*- coding: utf-8 -*-
# @Author : Young
# @Time : 17:19  2022-05-20
# from model.model import Unet
import numpy as np

from model.Unet import Unet
from model.RcgNet import Recg
from dataloader import dataloder, carNumloder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import gc
from tqdm import tqdm


def segTrain_fit(batchsize=None, epoch=None, imagsize=None, device=None):
    model = Unet(class_num=2)
    model = torch.load("segmodel_best.pt", map_location="cuda")
    traindata = dataloder(imagesize=imagsize, dirpath=r"D:\MyNAS\CarPlate\dataset\segtrain\image", number=None)
    trainloader = DataLoader(traindata, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=4)

    evaldata = dataloder(imagesize=imagsize, dirpath=r"D:\MyNAS\CarPlate\dataset\test\image", number=None)
    evalloader = DataLoader(evaldata, batch_size=batchsize, shuffle=True,
                            drop_last=True, num_workers=2)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1e-8, momentum=0.9)
    lossFunc = torch.nn.CrossEntropyLoss()

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    model.to(device)
    model.train()
    bestloss = np.Inf
    eval_step = int(0.5 * len(trainloader))
    for e_step in range(epoch):
        for step, batch in enumerate(trainloader):
            img, mask = batch
            img = img.to(device).float()
            mask = mask.to(device).long()
            pred = model(img)
            loss = lossFunc(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print("epoch : %d / %d , step : %d / %d , loss : %.4f" % (
                    e_step, epoch, step, len(trainloader), loss.item()))
            if step % eval_step == 0:
                model.eval()
                torch.save(model, "segmodel.pt")
                bestloss = evalModel(model, evalloader, lossFunc, bestloss, device="cuda", model_name="segmodel")
                model.train()
        torch.save(model, "segmodel.pt")
        lr_scheduler.step()


def RecgTrain_fit(imagsize:tuple,batchsize=None, epoch=None,  device=None):
    model = Recg(image_size=imagsize)
    # model = torch.load("recg_best.pt", map_location="cuda")
    traindata = carNumloder(imagesize=imagsize, dirpath=r"D:\MyNAS\CarPlate\dataset\recgtrain\image1", number=None)
    imgList = traindata.ImageList
    train_ratio = 0.95
    train_num = int(train_ratio * len(imgList))

    train_imgList = imgList[:train_num]
    eval_imgList = imgList[train_num:]

    print("train image size : ", len(train_imgList))
    print("eval image size : ", len(eval_imgList))

    traindata = carNumloder(imagesize=imagsize, ImageList=train_imgList, number=None)
    trainloader = DataLoader(traindata, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=4)

    evaldata = carNumloder(imagesize=imagsize, ImageList=eval_imgList, number=None)
    evalloader = DataLoader(evaldata, batch_size=batchsize, shuffle=True,
                            drop_last=True, num_workers=2)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
    lossFunc = torch.nn.CrossEntropyLoss(ignore_index=255)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    model.to(device)
    model.train()
    bestloss = np.Inf
    eval_step = int(0.5 * len(trainloader))
    for e_step in range(epoch):
        for step, batch in enumerate(trainloader):
            img, mask = batch
            img = img.to(device).float()
            mask = mask.to(device).long()
            pred = model(img)
            loss = lossFunc(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                print("epoch : %d / %d , step : %d / %d , loss : %.4f , lr : %.4f" % (
                    e_step, epoch, step, len(trainloader), loss.item(), lr))
            if step % eval_step == 0:
                model.eval()
                torch.save(model, "recg.pt")
                bestloss = evalModel(model, evalloader, lossFunc, bestloss, device="cuda", model_name="recg")
                model.train()
        torch.save(model, "recg.pt")
        lr_scheduler.step()


def evalModel(model, eloader, lossfunction, bestloss, device="cuda", model_name=None):
    model.eval()
    loss = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(eloader)):
            img, mask = batch
            img = img.to(device).float()
            mask = mask.to(device).long()
            pred = model(img)
            loss += lossfunction(pred, mask)
        loss /= len(eloader)

        if bestloss > loss:
            print("eval loss improved from %.4f to %.4f" % (bestloss, loss))
            bestloss = loss
            torch.save(model, "{}_best.pt".format(model_name))
        else:
            print("eval loss dont improved bestloss : %.4f , eval loss :  %.4f" % (bestloss, loss))

        del img, batch, mask, pred, model
        gc.collect()
        torch.cuda.empty_cache()
    return bestloss


if __name__ == '__main__':
    # batchsize = 9
    # epoch = 50
    # imagsize = 512
    device = "cuda"
    # segTrain_fit(batchsize=batchsize, epoch=epoch, imagsize=imagsize, device=device)

    batchsize = 256
    epoch = 500
    imagsize = (64, 192)
    RecgTrain_fit(imagsize=imagsize, batchsize=batchsize, epoch=epoch, device=device)