import torch
import numpy as np
import cv2
import sys
import os
import shutil
from torch.utils.data import DataLoader
import torch.distributed as dist
from dataset import DFDataset, TripletDFDataset2,AdaptiveDFDataset
from torchsummary import summary
from net import TripletNet,SimpleNet, IntraTripletNet
from torch import nn, optim
import net_conf as config
from torchvision import models
from tqdm import tqdm
from strong_transform import augmentation, trans
from efficientnet_pytorch import EfficientNet
from prefetch_generator import BackgroundGenerator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import segmentation_models_pytorch as smp
from sklearn.manifold import TSNE 
gpu = sys.argv[1]
jsonpath = sys.argv[2]
ckptname = jsonpath.split('/')[-1][:-5]
print(ckptname)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

if not os.path.exists(config.save_dir):
    os.makedirs(config.save_dir)


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


def load_model(model, path):
    ckpt = torch.load(path, map_location="cpu")
    # print(ckpt)
    model.load_state_dict(ckpt["state_dict"])
    return model

unet1=buildmodel().cuda()
unet2=buildmodel().cuda()
unet3=buildmodel().cuda()
unet4=buildmodel().cuda()
gunet=buildmodel().cuda()
modelpaths='/Data/olddata_D/ypp/triplet/iccv/u-efficientnet-b0/rawbest/'
modellist=os.listdir(modelpaths)

modellist.sort()
print(modellist)
dfunet=load_model(unet1,modelpaths+modellist[0])
f2funet=load_model(unet2,modelpaths+modellist[1])
fsunet=load_model(unet3,modelpaths+modellist[2])
ntunet=load_model(unet4,modelpaths+modellist[3])
# allnet=load_model(gunet,'/Data/olddata_D/ypp/triplet/iccv/u-efficientnet-b0/allbest/eff0_smp_unet_ff++_raw_all_df19_ckpt-0.1536421732328433.pth')
print('load model finish')

dfvalidate_dataset = AdaptiveDFDataset(
    config.data_path, 'test', trans=trans, augment=augmentation, jsonpath='json/ff++_raw_Deepfakes.json')

f2fvalidate_dataset = AdaptiveDFDataset(
    config.data_path, 'test', trans=trans, augment=augmentation, jsonpath='json/ff++_raw_Face2Face.json')

fsvalidate_dataset = AdaptiveDFDataset(
    config.data_path, 'test', trans=trans, augment=augmentation, jsonpath='json/ff++_raw_FaceSwap.json')

ntvalidate_dataset = AdaptiveDFDataset(
    config.data_path, 'test', trans=trans, augment=augmentation, jsonpath='json/ff++_raw_NeuralTextures.json')




def extract(model,validate_dataset):
    model = model.cuda()
    model = nn.DataParallel(model)
    validate_loader = DataLoaderX(
        validate_dataset, batch_size=config.batch_size, num_workers=config.workers, pin_memory=True)

    test_embeddings = []
    test_labels     = []
    with tqdm(validate_loader, desc = 'Plot' ) as bar:
        for b, batch in enumerate( bar ):
            anchorimg, label,maskimg,_ = batch
            anchorimg = anchorimg.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            maskimg = maskimg.cuda(non_blocking=True)

            afeature, anchor, label = model(anchorimg)

            embeddings = afeature.detach( ).cpu( ).numpy( )
            # print(embeddings.shape)
            label = label.detach( ).cpu().numpy()
            embeddings=embeddings.reshape((embeddings.shape[0],embeddings.shape[1]))
            test_embeddings.append( embeddings )
            test_labels.append( label )

    test_embeddings=np.array(test_embeddings)
    test_labels=np.array(test_labels)
    print(test_embeddings.shape)
    # np.save('test_embeddings.npy',test_embeddings)
    # np.save('test_labels.npy',test_labels)
    return test_embeddings,test_labels

def vis(embeddings,labels,imgname):
    count=0
    np.save(imgname+'+test_embeddings.npy',embeddings)
    for i in range(embeddings.shape[0]):
        t1=embeddings[i]
        count+=t1.shape[0]

    feature=np.zeros((count,320))
    label=np.zeros((count))
    # print(feature.shape,label.shape)
    index=0
    for i in range(embeddings.shape[0]):
        t1=embeddings[i]
        t2=labels[i].argmax(axis=1)
        print(t1.shape,t2.shape)
        feature[index:index+t1.shape[0]]=t1
        label[index:index+t1.shape[0]]=t2
        index+=t1.shape[0]
    print(feature.shape,label.shape)
    print(label)
    print(label[label==1].shape)
    # feature=feature[:100,:]
    # label=label[:100]
    # std=StandardScaler()
    # feature=std.fit_transform(feature)
    tsne = TSNE(n_components=2)
    tsne.fit_transform(feature) #进行数据降维,并返回结果
    t2d_vector = tsne.embedding_
    # print(t2d_vector.shape)
    # pca=PCA(n_components=2)
    # t2d_vector=pca.fit_transform(feature)
    real=t2d_vector[label==0]
    fake=t2d_vector[label==1]
    # print(real,fake)
    print(real.shape,fake.shape)
    # print(real[:,0],real[1,:])
    plt.clf()
    plt.scatter(real[:,0], real[:,1], c='b')
    plt.scatter(fake[:,0], fake[:,1], c='r')
    plt.savefig(imgname)



dfembeddings,labels=extract(dfunet,dfvalidate_dataset)
vis(dfembeddings,labels,'df')
fsembeddings,labels=extract(fsunet,fsvalidate_dataset)
vis(fsembeddings,labels,'fs')
f2fembeddings,labels=extract(f2funet,f2fvalidate_dataset)
vis(f2fembeddings,labels,'f2f')
ntembeddings,labels=extract(ntunet,ntvalidate_dataset)
vis(ntembeddings,labels,'nt')
# np.save('npy/df_real_all.npy',dfembeddings)
# np.save('npy/fs_real_all.npy',fsembeddings) 
# np.save('npy/f2f_real_all.npy',f2fembeddings)
# np.save('npy/nt_real_all.npy',ntembeddings)