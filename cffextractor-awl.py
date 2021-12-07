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
from dataset import AdaptiveDFDataset2,CelebDFAdaptiveDFDataset
from torchsummary import summary
from net import TripletNet, SimpleNet
from torch import nn, optim
import net_conf as config
from torchvision import models
from tqdm import tqdm
import net
import random
from strong_transform import augmentation, trans
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import roc_auc_score
import segmentation_models_pytorch as smp

print(sys.argv)
stdout_backup = sys.stdout
# define the log file that receives your log info
para=sys.argv[3]+sys.argv[4]+sys.argv[5]+sys.argv[6]
log_file = open("logs/awl-"+sys.argv[0][:-3]+para+"c23_message-0.log", "w")
# redirect print output to log file
sys.stdout = log_file


# gpu = sys.argv[1]
jsonpath = sys.argv[1]
ckptname = jsonpath.split('/')[-1][:-5]
print(ckptname)

a = float(sys.argv[2])
b = float(sys.argv[3])
c = float(sys.argv[4])
d = float(sys.argv[5])

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

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

    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_data is not None:
            data, label = self.next_data
            self.preload()
            return data, label
        else:
            return None, None


def save_checkpoint(path, state_dict, epoch=0, arch="", acc1=0):
    filedir = os.path.dirname(path)
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
    aux_params = dict(
        pooling='avg',             # one of 'avg', 'max'
        dropout=0.5,               # dropout ratio, default is None
        activation='sigmoid',      # activation function, default is None
    )

    unet = smp.Unet(
        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_name="efficientnet-b0",
        # use `imagenet` pretrained weights for encoder initialization
        encoder_weights="imagenet",
        classes=1,
        activation='sigmoid',
        aux_params=aux_params
    )
    return unet

def generatelabel(catelabel):
    label = torch.ones(catelabel.shape)
    index = torch.argmax(catelabel, dim=1)
    for i in range(label.shape[0]):
        label[i][index[i]]=0
    return label.cuda()


def generatedislabel(catelabel):
    labels = []
    for i in range(catelabel.shape[0]):
        if catelabel[i] == 0:
            l = [1, 0, 0, 0]
        if catelabel[i] == 1:
            l = [0, 1, 0, 0]
        if catelabel[i] == 2:
            l = [0, 0, 1, 0]
        if catelabel[i] == 3:
            l = [0, 0, 0, 1]
        labels.append(l)
    labels = torch.tensor(labels)
    return labels.cuda()


def processlist(anchorlist):
    anchorimg, label, maskimg, catelabel = anchorlist
    anchorimg = anchorimg.cuda(non_blocking=True)
    label = label.cuda(non_blocking=True)
    maskimg = maskimg.cuda(non_blocking=True)
    catelabel = catelabel.cuda(non_blocking=True)
    return anchorimg, label, maskimg, catelabel


def calcLoss(a, df, f2f, fs, nt, label):
    loss = 0
    # label = torch.argmax(label, dim=-1)
    for i in range(label.shape[0]):
        if label[i] == 0:
            l = nn.MSELoss()(a[i], df[i])
        elif label[i] == 1:
            l = nn.MSELoss()(a[i], f2f[i])
        elif label[i]==2:
            l=nn.MSELoss()(a[i],fs[i])
        elif label[i] == 3:
            l = nn.MSELoss()(a[i], nt[i])
        loss += l
    return loss/label.shape[0]

def generatorloss(catepred,catelabel):
    index = torch.argmax(catelabel, dim=-1)
    label = torch.zeros(catelabel.shape)
    ls=[]
    criterion=nn.BCELoss()
    for i in range(label.shape[0]):
        l=[0,1,2,3]
        print(index[i])
        l=list(set(l)-set([index[i].item()]))
        ls.append(l)
    ls=torch.tensor(ls)
    print(ls)
    label1 = torch.zeros(catelabel.shape)
    for i in range(label1.shape[0]):
        label1[i,ls[i,0]]=1

    label2 = torch.zeros(catelabel.shape)
    for i in range(label2.shape[0]):
        label2[i,ls[i,1]]=1

    label3 = torch.zeros(catelabel.shape)
    for i in range(label3.shape[0]):
        label3[i,ls[i,2]]=1
    loss=criterion
    return label.cuda()


unet1 = buildmodel().cuda()
unet2 = buildmodel().cuda()
unet3 = buildmodel().cuda()
unet4 = buildmodel().cuda()
gunet = buildmodel()
modelpaths = 'saved_models/bestmodel/c23best10/'
modellist = os.listdir(modelpaths)
modellist.sort()
dfunet = load_model(unet1, modelpaths+modellist[0])
f2funet = load_model(unet2, modelpaths+modellist[1])
fsunet = load_model(unet3, modelpaths+modellist[2])
ntunet = load_model(unet4, modelpaths+modellist[3])
print('load model finish')
modelname = gunet.name

gunet = gunet.cuda()
gunet = nn.DataParallel(gunet)
disnet = net.Discrimintor2().cuda()
criterion = smp.utils.losses.DiceLoss()

criterion2 = nn.BCELoss()
bloss = nn.BCELoss()
awl=AutomaticWeightedLoss(4)

gunet_optimizer = optim.Adam([
                {'params': gunet.parameters()},
                {'params': awl.parameters()}
            ], lr=1e-5, weight_decay= 1e-6)
disnet_optimizer = optim.Adam(disnet.parameters(), 1e-5, weight_decay=1e-6)

scheduler = optim.lr_scheduler.StepLR(gunet_optimizer, step_size=1, gamma=0.2)
train_dataset = AdaptiveDFDataset2(
    config.data_path, 'train', trans=trans, augment=augmentation, jsonpath=jsonpath, exceptdata=None)

validate_dataset1 = AdaptiveDFDataset2(
    config.data_path, 'test', trans=trans, augment=augmentation, jsonpath=jsonpath, exceptdata=None)

validate_dataset2 = CelebDFAdaptiveDFDataset(
    config.data_path, 'test', trans=trans, augment=augmentation, jsonpath='json/celebdf.json')
validate_dataset3 = CelebDFAdaptiveDFDataset(
    config.data_path, 'test', trans=trans, augment=augmentation, jsonpath='json/dfdc.json')

train_loader = DataLoaderX(train_dataset, batch_size=25,
                           num_workers=config.workers, pin_memory=True)
validate_loader = DataLoaderX(
    validate_dataset1, batch_size=25, num_workers=config.workers, pin_memory=True)

validate_loader2 = DataLoaderX(
    validate_dataset2, batch_size=25, num_workers=config.workers, pin_memory=True)

validate_loader3 = DataLoaderX(
    validate_dataset3, batch_size=25, num_workers=config.workers, pin_memory=True)

epochs=10
best_loss = 100
best_epoch = 0
best_model = None
start_epoch = 0
for epoch in range(epochs):
    gunet.train()
    dfunet.eval()
    f2funet.eval()
    fsunet.eval()
    ntunet.eval()
    disnet.train()

    train_loss = []
    train_acc = []
    train_auc = []
    labels=[]
    preds=[]
    with tqdm(train_loader, desc='Batch') as bar:
        count = 0
        for bi, batch in enumerate(bar):
            anchorimg, label, maskimg, catelabel = batch
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
            catepred = disnet(afeature)
            cateloss = nn.CrossEntropyLoss()(catepred.float(), catelabel.long())
            cateloss.backward(retain_graph=True)
            disnet_optimizer.step()

            gunet_optimizer.zero_grad()
            predictloss = criterion2(pred, label)
            maskloss = criterion(anchor, maskimg)
            catepred = disnet(afeature)
            gcateloss = generatorloss(catepred.float(), catelabel.float().cuda())
            mseloss = calcLoss(afeature, dfafeature, f2fafeature,
                               fsafeature, ntafeature, catelabel)
            # loss = a * predictloss + b * maskloss + c * mseloss + d * gcateloss
            loss = awl(predictloss,maskloss,gcateloss,mseloss)
            loss.backward()
            gunet_optimizer.step()
            pred=pred.detach().cpu()
            label=label.cpu()
            labels.append(label)
            preds.append(pred)
            try:
                auc = roc_auc_score(label, pred)
            except:
                auc = 0.5
            out = torch.argmax(pred.data, 1)
            label = torch.argmax(label.data, 1)
            batch_acc = torch.sum(out == label).item() / len(out)
            batch_loss = loss.item()

            bar.set_postfix(
                cateloss=cateloss.item(),
                gcateloss=gcateloss.item(),
                maskloss=maskloss.item(),
                mseloss=mseloss.item(),
                predictloss=predictloss.item(),
                batch_loss=batch_loss,
                batch_acc=batch_acc,
                auc=auc
            )
            train_loss.append(batch_loss)
            train_acc.append(batch_acc)
            train_auc.append(auc)
        labels=torch.cat(labels,dim=0)
        preds=torch.cat(preds,dim=0)
        epoch_auc=roc_auc_score(labels, preds)
        print(awl.parameters())
    epoch_loss = np.mean(train_loss)
    epoch_acc = np.mean(train_acc)
    print(epoch, "Train Epoch Loss:", epoch_loss, "Train Epoch Acc:",
          epoch_acc, "Train Epoch Auc:", epoch_auc)
    torch.cuda.empty_cache()
    for b, batch in enumerate(awl.parameters()):
        print(b,batch)

    gunet.eval()
    val_loss = []
    val_acc = []
    val_auc = []
    labels=[]
    preds=[]
    best_loss = 100
    with tqdm(validate_loader, desc='Batch') as bar:
        for b, batch in enumerate(bar):
            anchorimg, label, maskimg, catelabel = batch
            anchorimg = anchorimg.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            afeature, anchor, pred = gunet(anchorimg)
            predictloss = criterion2(pred, label)
            pred=pred.detach().cpu()
            label=label.cpu()
            try:
                auc = roc_auc_score(label, pred)
            except:
                auc = 0.5
            
            labels.append(label)
            preds.append(pred)
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
    labels=torch.cat(labels,dim=0)
    preds=torch.cat(preds,dim=0)
    epoch_auc=roc_auc_score(labels, preds)
    epoch_loss = np.mean(val_loss)
    epoch_acc = np.mean(val_acc)
    print(epoch, "FF++ Val Epoch Loss:", epoch_loss, "Val Epoch Acc:",
          epoch_acc, "Val Epoch Auc:", epoch_auc)
    torch.cuda.empty_cache()

    val_loss = []
    val_acc = []
    val_auc = []
    labels=[]
    preds=[]
    best_loss = 100
    with tqdm(validate_loader2, desc='Batch') as bar:
        for b, batch in enumerate(bar):
            # anchorimg, label, maskimg, catelabel = batch
            anchorimg, label = batch
            anchorimg = anchorimg.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            afeature, anchor, pred = gunet(anchorimg)
            predictloss = criterion2(pred, label)
            pred=pred.detach().cpu()
            label=label.cpu()
            try:
                auc = roc_auc_score(label, pred)
            except:
                auc = 0.5
            
            labels.append(label)
            preds.append(pred)
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
    labels=torch.cat(labels,dim=0)
    preds=torch.cat(preds,dim=0)
    epoch_auc=roc_auc_score(labels, preds)
    epoch_loss = np.mean(val_loss)
    epoch_acc = np.mean(val_acc)
    print(epoch, "CelebDF Val Epoch Loss2:", epoch_loss, "Val Epoch Acc2:",
          epoch_acc, "Val Epoch Auc2:", epoch_auc)
    
    val_loss = []
    val_acc = []
    val_auc = []
    labels=[]
    preds=[]
    best_loss = 100
    with tqdm(validate_loader3, desc='Batch') as bar:
        for b, batch in enumerate(bar):
            # anchorimg, label, maskimg, catelabel = batch
            anchorimg, label = batch
            anchorimg = anchorimg.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            afeature, anchor, pred = gunet(anchorimg)
            predictloss = criterion2(pred, label)
            pred=pred.detach().cpu()
            label=label.cpu()
            try:
                auc = roc_auc_score(label, pred)
            except:
                auc = 0.5
            
            labels.append(label)
            preds.append(pred)
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
    labels=torch.cat(labels,dim=0)
    preds=torch.cat(preds,dim=0)
    epoch_auc=roc_auc_score(labels, preds)
    epoch_loss = np.mean(val_loss)
    epoch_acc = np.mean(val_acc)
    print(epoch, "DFDC Val Epoch Loss2:", epoch_loss, "Val Epoch Acc2:",
          epoch_acc, "Val Epoch Auc2:", epoch_auc)
log_file.close()
