import torch
import numpy as np
import cv2
import sys
import os
import shutil
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import torch.distributed as dist
from dataset import *
from torchsummary import summary
from net import TripletNet, SimpleNet
from torch import nn, optim
import net_conf as config
from torchvision import models
from tqdm import tqdm
import net
import cv2
from strong_transform import augmentation, trans
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import roc_auc_score
import segmentation_models_pytorch as smp

# gpu = sys.argv[1]
jsonpath = sys.argv[1]

ckptname = jsonpath.split('/')[-1][:-5]
print(ckptname)



class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
            print(self.next_data)
        except StopIteration:
            self.next_data = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)

    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_data is not None:
            data, label = self.next_data
            self.preload()
            return data, label
        else:
            return None, None

def save_checkpoint(path, state_dict, epoch=0, arch="", acc1=0):
    filedir=os.path.dirname(path)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if torch.is_tensor(v):
            v = v.cpu()
        new_state_dict[k] = v

    torch.save({
        "epoch": epoch,
        "arch": arch,
        "acc1": acc1,
        "state_dict": new_state_dict,
    }, path)


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
               # define number of output labels
    )
    # unet = EfficientUnet(encoder, out_channels=1, concat_input=True)
    # unet=UNet(3,1)
    unet = smp.Unet(
        encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization                    
        classes=1,
        activation='sigmoid',
        aux_params=aux_params
    )
    return unet

def calcLoss(a,df,f2f,fs,nt,label):
    loss=0
    label=torch.argmax(label,dim=-1)
    for i in range(label.shape[0]):
        if label[i]==0:
            l=nn.MSELoss()(a[i],df[i])
        elif label[i]==1:
            l=nn.MSELoss()(a[i],f2f[i])
        elif label[i]==2:
            l=nn.MSELoss()(a[i],fs[i])
        elif label[i]==3:
            l=nn.MSELoss()(a[i],nt[i])
        loss+=l
    return loss/label.shape[0]
        

gunet=buildmodel()
# path='/Data/olddata_D/ypp/triplet/iccv/u-efficientnet-b0/eff0_smp_unet_ff++_c23_all18_ckpt-0.35373437756084464.pth'
# path='/Data/olddata_D/ypp/triplet/iccv/u-efficientnet-b0/eff0_smp_unet_ff++_c23_all19_ckpt-0.38280598784444886.pth'
# path='saved_models/eff0_smp_adaptiveunet_Noneff++_c23_all18_ckpt-0.3782317639644176.pth'
path='saved_models/u-efficientnet-b0/segex_3_eff0_smp_unet_ff++_raw_all0_ckpt-1.986181451376102.pth'
# path='saved_models/eff0_smp_unet2_ff++_c40_all_None_17_ckpt-0.509520394196472.pth'
gunet=load_model(gunet,path)
print('load model finish')
modelname=gunet.name
# summary(unet,input_size=(3,224, 224))
# print()
gunet = gunet.cuda()
gunet = nn.DataParallel(gunet)
criterion = smp.utils.losses.DiceLoss()

criterion2 = nn.BCELoss()

validate_dataset = CelebDFAdaptiveDFDataset(
    config.data_path, 'test', trans=trans, augment=augmentation, jsonpath=jsonpath)

validate_loader = DataLoaderX(
    validate_dataset, batch_size=12, num_workers=config.workers, pin_memory=True)

gunet.eval()
val_loss = []
val_acc = []
val_auc=[]
val_maskloss=[]
best_loss = 1
with tqdm(validate_loader, desc='Batch') as bar:
    for b, batch in enumerate(bar):
        anchorimg, label = batch
        anchorimg = anchorimg.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        # maskimg = maskimg.cuda(non_blocking=True)
        # catelabel = catelabel.cuda(non_blocking=True)

        afeature, anchor, pred = gunet(anchorimg)
        # maskloss = criterion(anchor, maskimg)
        predictloss = criterion2(pred, label)
        try:
            auc=roc_auc_score(label.cpu(),pred.detach().cpu())
        except:
            auc=0.5
        out = torch.argmax(pred.data, 1)
        label = torch.argmax(label.data, 1)
        batch_acc = torch.sum(out == label).item() / len(out)
        
        loss = predictloss
        batch_loss = loss.item()
        bar.set_postfix(
            batch_loss=batch_loss,
            predictloss=predictloss.item(),
            batch_acc=batch_acc,
            auc=auc
        )
        val_loss.append(batch_loss)
        val_acc.append(batch_acc)
        val_auc.append(auc)
epoch_loss = np.mean(val_loss)
epoch_acc = np.mean(val_acc)
epoch_auc=np.mean(val_auc)
print(ckptname,"Val Epoch Loss:", epoch_loss, "Val Epoch Acc:", epoch_acc, "Val Epoch Auc:", epoch_auc)

