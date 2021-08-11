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
from dataset import DFDataset, TripletDFDataset3,AdaptiveDFDataset
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
exceptdata=None
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
        classes=2,                 # define number of output labels
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

def generatelabel(catelabel):
    index=torch.argmax(catelabel,dim=1)
    i0=[1,0,0,0]
    i1=[0,1,0,0]
    i2=[0,0,1,0]
    i3=[0,0,0,1]
    labels1=[]
    labels2=[]
    labels3=[]
    for i in range(index.shape[0]):
        if index[i]==0:
            labels1.append(i1)
            labels2.append(i2)
            labels3.append(i3)
        if index[i]==1:
            labels1.append(i0)
            labels2.append(i2)
            labels3.append(i3)
        if index[i]==2:
            labels1.append(i0)
            labels2.append(i1)
            labels3.append(i3)
        if index[i]==3:
            labels1.append(i0)
            labels2.append(i1)
            labels3.append(i2)
    labels1t=torch.Tensor(labels1)
    labels2t=torch.Tensor(labels2)
    labels3t=torch.Tensor(labels3)
    return labels1t,labels2t,labels3t

        
            
        

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
        


unet1=buildmodel().cuda()
unet2=buildmodel().cuda()
unet3=buildmodel().cuda()
unet4=buildmodel().cuda()
gunet=buildmodel()
modelpaths='/data1/triplet/bestmodel/c23best/'
modellist=os.listdir(modelpaths)
modellist.sort()
dfunet=load_model(unet1,modelpaths+modellist[0])
f2funet=load_model(unet2,modelpaths+modellist[1])
fsunet=load_model(unet3,modelpaths+modellist[2])
ntunet=load_model(unet4,modelpaths+modellist[3])
print('load model finish')
modelname=gunet.name
# summary(unet,input_size=(3,224, 224))
# print()
gunet = gunet.cuda()
gunet = nn.DataParallel(gunet)
disnet=net.Discrimintor().cuda()
disnet= nn.DataParallel(disnet)
criterion = smp.utils.losses.DiceLoss()

criterion2 = nn.BCELoss()
# encoder_optimizer = optim.SGD(encoder.parameters(), 1e-4,momentum=0.9)
gunet_optimizer = optim.SGD(gunet.parameters(), 1e-4,
                           momentum=0.9)
disnet_optimizer = optim.SGD(disnet.parameters(), 1e-4,
                           momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(gunet_optimizer, step_size=1, gamma=0.2)
train_dataset = AdaptiveDFDataset(
    config.data_path, 'train', trans=trans, augment=augmentation, jsonpath=jsonpath,exceptdata=exceptdata)
validate_dataset = AdaptiveDFDataset(
    config.data_path, 'test', trans=trans, augment=augmentation, jsonpath=jsonpath,exceptdata=exceptdata)


train_loader = DataLoaderX(train_dataset, batch_size=config.batch_size,
                           num_workers=config.workers, pin_memory=True)
validate_loader = DataLoaderX(
    validate_dataset, batch_size=config.batch_size, num_workers=config.workers, pin_memory=True)

best_loss = 100
best_epoch = 0
best_model = None
start_epoch = 0
for epoch in range(config.epochs):
    if epoch < start_epoch:
        scheduler.step()
        continue
    gunet.train()
    dfunet.eval()
    f2funet.eval()
    fsunet.eval()
    ntunet.eval()
    disnet.train()
    train_loss = []
    train_acc = []
    train_auc = []
    with tqdm(train_loader, desc='Batch') as bar:
        count=0
        for b, batch in enumerate(bar):
            anchorimg, label,maskimg,catelabel = batch
            anchorimg = anchorimg.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            maskimg = maskimg.cuda(non_blocking=True)
            catelabel = catelabel.cuda(non_blocking=True)
            
            afeature, anchor, pred = gunet(anchorimg)
            dfafeature, dfanchor, dfpred = dfunet(anchorimg)
            f2fafeature, f2fanchor, f2fpred = f2funet(anchorimg)
            fsafeature, fsanchor, fspred = fsunet(anchorimg)
            ntafeature, ntanchor, ntpred = ntunet(anchorimg)

            disnet_optimizer.zero_grad()
            
            # catenonehotlabel=get_one_hot(catelabel, 4)
            catepred=disnet(afeature)
            cateloss=nn.BCELoss()(catepred.float(),catelabel.float())
            cateloss.backward(retain_graph=True)
            disnet_optimizer.step()

            gunet_optimizer.zero_grad()
            mseloss=calcLoss(afeature,dfafeature,f2fafeature,fsafeature,ntafeature,catelabel)
            predictloss=criterion2(pred,label)
            maskloss = criterion(anchor, maskimg)
            catepred=disnet(afeature)
            labels1t,labels2t,labels3t=generatelabel(catelabel.float())
            gcateloss1= nn.BCELoss()(catepred.float(),labels1t.float().cuda())
            gcateloss2= nn.BCELoss()(catepred.float(),labels2t.float().cuda())
            gcateloss3= nn.BCELoss()(catepred.float(),labels3t.float().cuda())
            gcateloss = gcateloss1+gcateloss2+gcateloss3
            loss = mseloss + predictloss + maskloss+gcateloss
            loss.backward()
            gunet_optimizer.step()
            try:
                auc=roc_auc_score(label.cpu(),pred.detach().cpu())
            except:
                auc=0.5
            out = torch.argmax(pred.data, 1)
            label = torch.argmax(label.data, 1)
            batch_acc = torch.sum(out == label).item() / len(out)
            batch_loss = loss.item()
            
            bar.set_postfix(
                cateloss=cateloss.item(),
                gcateloss = gcateloss.item(),
                mseloss=mseloss.item(),
                maskloss=maskloss.item(),
                predictloss=predictloss.item(),
                batch_loss=batch_loss,
                batch_acc=batch_acc,
                auc=auc
            )
            train_loss.append(batch_loss)
            train_acc.append(batch_acc)
            train_auc.append(auc)
    epoch_loss = np.mean(train_loss)
    epoch_acc = np.mean(train_acc)
    epoch_auc = np.mean(train_auc)
    print(epoch, "Train Epoch Loss:", epoch_loss, "Train Epoch Acc:", epoch_acc, "Train Epoch Auc:", epoch_auc)

    gunet.eval()
    val_loss = []
    val_acc = []
    val_auc=[]
    best_loss = 1
    with tqdm(validate_loader, desc='Batch') as bar:
        for b, batch in enumerate(bar):
            anchorimg, label,maskimg,catelabel = batch
            anchorimg = anchorimg.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            maskimg = maskimg.cuda(non_blocking=True)
            catelabel = catelabel.cuda(non_blocking=True)

            afeature, anchor, pred = gunet(anchorimg)
            maskloss = criterion(anchor, maskimg)
            predictloss = criterion2(pred, label)
            try:
                auc=roc_auc_score(label.cpu(),pred.detach().cpu())
            except:
                auc=0.5
            out = torch.argmax(pred.data, 1)
            label = torch.argmax(label.data, 1)
            batch_acc = torch.sum(out == label).item() / len(out)
            
            loss = maskloss+predictloss
            batch_loss = loss.item()
            bar.set_postfix(
                batch_loss=batch_loss,
                maskloss=maskloss.item(),
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
    print(epoch, "Val Epoch Loss:", epoch_loss, "Val Epoch Acc:", epoch_acc, "Val Epoch Auc:", epoch_auc)
    # if epoch_loss < best_loss:
    #     best_model=model
    #     best_loss=epoch_loss
    #     best_epoch=epoch
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        ckpt_path = os.path.join(config.save_dir, modelname+'new/eff0_smp_adaptiveunet_' + str(exceptdata)+
                                 ckptname+str(epoch)+"_ckpt-%s.pth" % (epoch_loss))
        save_checkpoint(
            ckpt_path,
            gunet.state_dict(),
            epoch=epoch + 1,
            acc1=0)
