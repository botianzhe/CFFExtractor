import torch
import numpy as np
import cv2
import sys
import os
import shutil
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torchvision import models
from tqdm import tqdm
import cv2
from strong_transform import  trans, augmentation
import segmentation_models_pytorch as smp

def load_model(model, path):
    ckpt = torch.load(path, map_location="cpu")
    # print(ckpt)
    start_epoch = ckpt.get("epoch", 0)
    best_acc = ckpt.get("acc1", 0.0)
    model.load_state_dict(ckpt["state_dict"])
    return model

def buildmodel():
    aux_params=dict(
        pooling='avg',             # one of 'avg', 'max'
        dropout=0.5,               # dropout ratio, default is None
        activation='sigmoid',      # activation function, default is None
        classes=2,                 # define number of output labels
    )

    unet = smp.Unet(
        encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization                    
        classes=1,
        activation='sigmoid',
        aux_params=aux_params
    )
    return unet

gunet=buildmodel()
path='checkpoint.pth'
gunet=load_model(gunet,path)
print('load model finish')

gunet = gunet.cuda()
gunet = nn.DataParallel(gunet)

gunet.eval()
anchorimg = cv2.imread('fake.jpg')
anchorimg = cv2.resize(anchorimg,(224,224))
anchorimg = trans(anchorimg).unsqueeze(0).cuda()
anchor, pred = gunet(anchorimg)
out = torch.argmax(pred.data, 1)
print(out)