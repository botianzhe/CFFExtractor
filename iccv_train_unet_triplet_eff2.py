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
from dataset import DFDataset, TripletDFDataset3
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

# import segmentation_models_pytorch as smp
from eval_kit.face_utils import norm_crop, FaceDetector
# from efficientunet import get_efficientunet_b0, EfficientNet, EfficientUnet
from model.unet import UNet
# gpu = sys.argv[1]
jsonpath = sys.argv[1]

ckptname = jsonpath.split('/')[-1][:-5]
print(ckptname)
# modeldir = '/Data/olddata_D/ypp/triplet/iccv/'
# minloss = 1000000
# minmodel = None
# for file in os.listdir(modeldir):
#     if ckptname in file:
#         loss = int(file.split('-')[1].split('.')[1][:6])
#         if loss < minloss:
#             minloss = loss
#             minmodel = file
# resume = modeldir+minmodel
# print(resume)
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu




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


# encoder = EfficientNet.encoder('efficientnet-b0',pretrained=True)
aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation='sigmoid',      # activation function, default is None
    classes=2,                 # define number of output labels
)
# unet = EfficientUnet(encoder, out_channels=1, concat_input=True)
# unet=UNet(3,1)
unet = smp.Unet(
    encoder_name="efficientnet-b3",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization                    
    classes=1,
    activation='sigmoid',
    aux_params=aux_params
)
modelname=unet.name
# summary(unet,input_size=(3,224, 224))
# print()
unet = unet.cuda()
unet = nn.DataParallel(unet)

criterion = smp.utils.losses.DiceLoss()
# criterion=nn.BCELoss()4

criterion2 = nn.BCELoss()
tripletloss = nn.TripletMarginLoss()
# encoder_optimizer = optim.SGD(encoder.parameters(), 1e-4,momentum=0.9)
unet_optimizer = optim.SGD(unet.parameters(), 1e-4,
                           momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(unet_optimizer, step_size=1, gamma=0.2)
train_dataset = TripletDFDataset3(
    config.data_path, 'train', trans=trans, augment=augmentation, jsonpath=jsonpath)
validate_dataset = TripletDFDataset3(
    config.data_path, 'test', trans=trans, augment=augmentation, jsonpath=jsonpath)


train_loader = DataLoaderX(train_dataset, batch_size=config.batch_size,
                           num_workers=config.workers, pin_memory=True)
validate_loader = DataLoaderX(
    validate_dataset, batch_size=config.batch_size, num_workers=config.workers, pin_memory=True)

best_loss = 1
best_epoch = 0
best_model = None
start_epoch = 0
for epoch in range(config.epochs):
    if epoch < start_epoch:
        scheduler.step()
        continue
    unet.train()
    train_loss = []
    train_acc = []
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
            # print(anchor)
            # print(afeature.shape,anchor.shape,pred.shape,maskimg.shape)
            # print()
            pfeature, _, _ = unet(positiveimg)
            nfeature, _, _ = unet(negativeimg)
            triplet_loss = tripletloss(afeature, pfeature, nfeature)
            # mask=anchor[0,:,:]
            # mask=mask.reshape((320,320,1))
            # cv2.imwrite('images/predmask'+str(count)+'.png',mask.detach().cpu().numpy()*255)
            maskloss = criterion(anchor, maskimg)
            predictloss = criterion2(pred, label)
            loss = maskloss + predictloss + triplet_loss
            # loss=maskloss
            loss.backward()
            unet_optimizer.step()
            # print(pred.shape)
            auc=roc_auc_score(label.cpu(),pred.detach().cpu())
            out = torch.argmax(pred.data, 1)
            label = torch.argmax(label.data, 1)
            batch_acc = torch.sum(out == label).item() / len(out)
            batch_loss = loss.item()
            
            bar.set_postfix(
                triplet_loss=triplet_loss.item(),
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

    unet.eval()
    val_loss = []
    val_acc = []
    val_auc=[]
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
        ckpt_path = os.path.join(config.save_dir, modelname+'/eff0_smp_unet_' +
                                 ckptname+str(epoch)+"_ckpt-%s.pth" % (epoch_loss))
        save_checkpoint(
            ckpt_path,
            unet.state_dict(),
            epoch=epoch + 1,
            acc1=0)
