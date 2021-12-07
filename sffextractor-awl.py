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
from dataset import TripletDFDataset3
from torchsummary import summary
from net import TripletNet, SimpleNet
from torch import nn, optim
import net_conf as config
from torchvision import models
from tqdm import tqdm
import cv2
from strong_transform import augmentation, trans
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import roc_auc_score
import segmentation_models_pytorch as smp
from utils.segmetric import SegmentationMetric

jsonpath = sys.argv[1]

ckptname = jsonpath.split('/')[-1][:-5]
print(ckptname)
aa = float(sys.argv[2])
bb = float(sys.argv[3])
cc = float(sys.argv[4])
# alpha = float(sys.argv[5])
alpha = float(sys.argv[5])
para=sys.argv[2]+sys.argv[3]+sys.argv[4] + sys.argv[5]
stdout_backup = sys.stdout
# define the log file that receives your log info
log_file = open("logs/1-"+sys.argv[0][:-3]+ckptname+para+"_message.log", "w")
# redirect print output to log file
sys.stdout = log_file


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

def calcmetrics(rev_wm,wmlabel):
    rev_wm=np.array(rev_wm>0.5,dtype=np.int64)
    wmlabel=np.array(wmlabel>0.5,dtype=np.int64)
    metric=SegmentationMetric(2)
    metric.addBatch(rev_wm, wmlabel)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    return pa, cpa, mpa, mIoU


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

aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation='sigmoid',      # activation function, default is None                # define number of output labels
)

unet = smp.Unet(
    encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization                    
    activation='sigmoid',
    aux_params=aux_params
)
modelname=unet.name

unet = unet.cuda()
unet = nn.DataParallel(unet)

criterion = smp.utils.losses.DiceLoss()
criterion2 = nn.BCELoss()

tripletloss = nn.TripletMarginLoss(margin=alpha)
ioumetric=smp.utils.metrics.IoU()
awl=AutomaticWeightedLoss(3)

unet_optimizer = optim.Adam([
                {'params': unet.parameters()},
                {'params': awl.parameters()}
            ], lr=1e-4, weight_decay= 1e-5)
scheduler = optim.lr_scheduler.StepLR(unet_optimizer, step_size=1, gamma=0.2)
train_dataset = TripletDFDataset3(
    config.data_path, 'train', trans=trans, augment=augmentation, jsonpath=jsonpath)
validate_dataset = TripletDFDataset3(
    config.data_path, 'test', trans=trans, augment=augmentation, jsonpath=jsonpath)

train_loader = DataLoaderX(train_dataset, batch_size=config.batch_size,
                           num_workers=config.workers, pin_memory=True)
validate_loader = DataLoaderX(
    validate_dataset, batch_size=config.batch_size, num_workers=config.workers, pin_memory=True)


epochs=10
best_loss = 1
best_epoch = 0
best_model = None
start_epoch = 0
for epoch in range(epochs):
    if epoch < start_epoch:
        scheduler.step()
        continue
    unet.train()
    train_loss = []
    train_acc = []
    train_iou=[]
    train_auc = []
    with tqdm(train_loader, desc='Batch') as bar:
        count=0
        for b, batch in enumerate(bar):
            anchorimg, positiveimg, negativeimg, maskimg, label = batch
            anchorimg = anchorimg.cuda(non_blocking=True)
            maskimg = maskimg.cuda(non_blocking=True)
            positiveimg = positiveimg.cuda(non_blocking=True)
            negativeimg = negativeimg.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            
            unet_optimizer.zero_grad()
            afeature, anchor, pred = unet(anchorimg)
            pfeature, _, _ = unet(positiveimg)
            nfeature, _, _ = unet(negativeimg)
            triplet_loss = tripletloss(afeature, pfeature, nfeature)
            maskloss = criterion(anchor, maskimg)
            predictloss = criterion2(pred, label)
            # loss = aa * predictloss  + bb * maskloss  + cc * triplet_loss
            loss = awl(predictloss,maskloss,triplet_loss)
            # loss=maskloss
            loss.backward()
            unet_optimizer.step()
            auc=roc_auc_score(label.cpu(),pred.detach().cpu())
            iou=calcmetrics(anchor.detach().cpu().numpy(),maskimg.cpu().numpy())[2]
            out = torch.argmax(pred.data, 1)
            label = torch.argmax(label.data, 1)
            batch_acc = torch.sum(out == label).item() / len(out)
            batch_loss = loss.item()
            bar.set_postfix(
                triplet_loss=triplet_loss.item(),
                maskloss=maskloss.item(),
                predictloss=predictloss.item(),
                batch_loss=batch_loss,
                batch_iou=iou,
                batch_acc=batch_acc,
                auc=auc
            )
            train_loss.append(batch_loss)
            train_acc.append(batch_acc)
            train_iou.append(iou)
            train_auc.append(auc)
    epoch_loss = np.mean(train_loss)
    epoch_acc = np.mean(train_acc)
    epoch_auc = np.mean(train_auc)
    epoch_iou= np.mean(train_iou)
    print(epoch, "Train Epoch Loss:", epoch_loss, "Train Epoch Acc:", epoch_acc, "Train Epoch Auc:", epoch_auc, "Train Epoch Iou:", epoch_iou)
    for b, batch in enumerate(awl.parameters()):
        print(b,batch)

    unet.eval()
    val_loss = []
    val_acc = []
    val_auc=[]
    val_iou=[]
    best_loss = 1
    with tqdm(validate_loader, desc='Batch') as bar:
        for b, batch in enumerate(bar):
            anchorimg, positiveimg, negativeimg, maskimg, label = batch
            anchorimg = anchorimg.cuda(non_blocking=True)
            maskimg = maskimg.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            _, anchor, pred = unet(anchorimg)
            maskloss = criterion(anchor, maskimg)
            predictloss = criterion2(pred, label)
            auc=roc_auc_score(label.cpu(),pred.detach().cpu())
            iou=calcmetrics(anchor.detach().cpu().numpy(),maskimg.cpu().numpy())[2]
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
                batch_iou=iou,
                auc=auc
            )
            val_loss.append(batch_loss)
            val_acc.append(batch_acc)
            val_auc.append(auc)
            val_iou.append(iou)
    epoch_loss = np.mean(val_loss)
    epoch_acc = np.mean(val_acc)
    epoch_auc=np.mean(val_auc)
    epoch_iou=np.mean(val_iou)
    print(epoch, "Val Epoch Loss:", epoch_loss, "Val Epoch Acc:", epoch_acc, "Val Epoch Auc:", epoch_auc, "Val Epoch Iou:", epoch_iou)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        ckpt_path = os.path.join(config.save_dir, modelname+'/1_smp_direct_' +
                                 ckptname+str(epoch)+ckptname+para+"_ckpt-%s-%s.pth" % (epoch_acc,epoch_auc))
        save_checkpoint(
            ckpt_path,
            unet.state_dict(),
            epoch=epoch + 1,
            acc1=0)
log_file.close()