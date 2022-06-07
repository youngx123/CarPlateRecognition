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
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
Scaler = GradScaler()

def segTrain_fit(batchsize=None, epoch=None, imagsize=None, device=None, modelname=None):
    model = Unet(class_num=1)
    # model = torch.load("segmodel_best.pt", map_location="cuda")
    traindata = dataloder(imagesize=imagsize, dirpath=r"D:\MyNAS\CarPlate\dataset\segtrain\image", number=10000)
    trainloader = DataLoader(traindata, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=4)

    evaldata = dataloder(imagesize=imagsize, dirpath=r"D:\MyNAS\CarPlate\dataset\test\image", number=None)
    evalloader = DataLoader(evaldata, batch_size=batchsize, shuffle=True,
                            drop_last=True, num_workers=2)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
    lossFunc = torch.nn.CrossEntropyLoss()
    lossFunc = torch.nn.BCEWithLogitsLoss()

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    model.to(device)
    model.train()
    bestloss = np.Inf
    eval_step = int(0.5 * len(trainloader))

    for e_step in range(epoch):
        for step, batch in enumerate(trainloader):
            img, mask = batch
            img = img.to(device).float()
            mask = mask.to(device).float()
            with autocast():
                pred = model(img)
                loss = lossFunc(pred, mask)
            Scaler.scale(loss).backward()
            Scaler.step(optimizer)
            Scaler.update()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            if step % 5 == 0:
                lr = optimizer.param_groups[0]['lr']
                print("epoch : %d / %d , step : %d / %d , loss : %.4f , lr : %.4f" % (
                    e_step, epoch, step, len(trainloader), loss.item(), lr))
            if step % eval_step == 0:
                model.eval()
                torch.save(model, "{}.pt".format(modelname))
                bestloss = evalModel(model, evalloader, lossFunc, bestloss, device="cuda", model_name=modelname)
                model.train()
        torch.save(model, "{}.pt".format(modelname))
        lr_scheduler.step()


def RecgTrain_fit(imagsize:tuple,batchsize=None, epoch=None,  device=None, modelName=None):
    ctc = True
    model = Recg(image_size=imagsize, ctc=ctc)
    model = torch.load("recg_best.pt", map_location="cuda")
    traindata = carNumloder(imagesize=imagsize, dirpath=r"D:\MyNAS\CarPlate\dataset\recgtrain\image1", number=None,ctc=ctc)
    imgList = traindata.ImageList
    train_ratio = 0.95
    train_num = int(train_ratio * len(imgList))

    train_imgList = imgList[:train_num]
    eval_imgList = imgList[train_num:]

    print("train image size : ", len(train_imgList))
    print("eval image size : ", len(eval_imgList))

    traindata = carNumloder(imagesize=imagsize, ImageList=train_imgList, number=None,ctc=ctc)
    trainloader = DataLoader(traindata, batch_size=batchsize, shuffle=True, drop_last=True, num_workers=4)

    evaldata = carNumloder(imagesize=imagsize, ImageList=eval_imgList, number=None,ctc=ctc)
    evalloader = DataLoader(evaldata, batch_size=batchsize, shuffle=True,
                            drop_last=True, num_workers=2)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
    if ctc:
        lossFunc = torch.nn.CTCLoss(blank=0)
    else:
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
            if ctc:
                t, b, c = pred.shape
                output = [t]*batchsize
                target_lenths = []
                new_mask = []
                for m in mask:
                    if 255 in m:
                        m = m[:-1]
                    new_mask.append(m)
                    target_lenths.append(len(m))

                mask = torch.cat(new_mask, 0).to(pred.device).long()
                output = torch.from_numpy(np.array(output)).to(pred.device).long()
                target_lenths = torch.from_numpy(np.array(target_lenths)).to(pred.device).long()
                loss = lossFunc(pred, mask, output, target_lenths)
            else:
                loss = lossFunc(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                lr = optimizer.param_groups[0]['lr']
                print("epoch : %d / %d , step : %d / %d , loss : %.4f , lr : %.4f" % (
                    e_step, epoch, step, len(trainloader), loss.item(), lr))
            if step % eval_step == 0:
                model.eval()
                torch.save(model, "{}.pt".format(modelName))
                bestloss = evalModel(model, evalloader, lossFunc, bestloss, device="cuda", model_name=modelName, ctc=ctc)
                model.train()
        torch.save(model, "{}.pt".format(modelName))
        lr_scheduler.step()


def evalModel(model, eloader, lossfunction, bestloss, device="cuda", model_name=None,ctc=None):
    model.eval()
    loss = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(eloader)):
            img, mask = batch
            img = img.to(device).float()
            mask = mask.to(device).float()
            pred = model(img)
            if ctc:
                t, b, c = pred.shape
                output = [t] * batchsize
                target_lenths = []
                new_mask = []
                for m in mask:
                    if 255 in m:
                        m = m[:-1]
                    new_mask.append(m)
                    target_lenths.append(len(m))

                mask = torch.cat(new_mask, 0).to(pred.device).long()
                output = torch.from_numpy(np.array(output)).to(pred.device).long()
                target_lenths = torch.from_numpy(np.array(target_lenths)).to(pred.device).long()
                loss += lossfunction(pred, mask, output, target_lenths)
            else:
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
    # batchsize = 16
    # epoch = 200
    # imagsize = 512
    device = "cuda"
    # segTrain_fit(batchsize=batchsize, epoch=epoch, imagsize=imagsize, device=device,modelname="sigmoidModel")

    batchsize = 256
    epoch = 500
    imagsize = (64, 192)
    RecgTrain_fit(imagsize=imagsize, batchsize=batchsize, epoch=epoch, device=device, modelName="recg")