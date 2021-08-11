import os
import cv2
import random
import json
from PIL import Image
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
# from cropout import cropelements
# import jpeg4py as jpeg


class DFDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.next_epoch()

    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = trainset
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            print(len(testset))
            self.dataset = testset
        self.dataset=self.dataset[:100]
        random.shuffle(self.dataset)

    def __getitem__(self, item):
        sample = self.dataset[item]
        # print(sample)
        anchor, positive, negative, label = sample
        anchorimg = cv2.imread(os.path.join(self.datapath, anchor))
        positiveimg = cv2.imread(os.path.join(self.datapath, positive))
        negativeimg = cv2.imread(os.path.join(self.datapath, negative))
        if self.dataselect == 'train':
            anchorimg = self.aug(image=anchorimg)['image']
            positiveimg = self.aug(image=positiveimg)['image']
            negativeimg = self.aug(image=negativeimg)['image']
        anchorimg = self.trans(anchorimg)
        positiveimg = self.trans(positiveimg)
        negativeimg = self.trans(negativeimg)
        data=[anchorimg, positiveimg, negativeimg]
        # data=np.array(data)
        data=torch.stack(data)
        return data, label

    def __len__(self):
        return len(self.dataset)

class TripletDFDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath,face_detector=None):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.face_detector=face_detector
        self.next_epoch()
        

    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = trainset
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            print(len(testset))
            self.dataset = testset

        # self.dataset=self.dataset[:100]    
        random.shuffle(self.dataset)
        self.labels=[l[1] for l in self.dataset]

    def __getitem__(self, item):
        sample,label = self.dataset[item]
        # print(os.path.join(self.datapath, sample))
        self.labels.append(label)
        anchorimg = cv2.imread(os.path.join(self.datapath, sample))
        anchorimg = cv2.resize(anchorimg,(299,299))
        if self.dataselect == 'train' and self.aug:
            anchorimg = self.aug(image=anchorimg)['image']
        anchorimg = self.trans(anchorimg)
        return anchorimg, label

    def __len__(self):
        return len(self.dataset)

    def getlabels(self):
        return self.labels


class TripletDFDataset2(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = trainset
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            print(len(testset))
            self.dataset = testset
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]  
        random.shuffle(self.dataset)
        self.labels=[l[1] for l in self.dataset]

    def __getitem__(self, item):
        sample,label = self.dataset[item]
        # print(os.path.join(self.datapath, sample))
        self.labels.append(label)

        anchorimg = cv2.imread(os.path.join(self.datapath, sample))
        # print(anchorimg.shape)
        anchorimg = cv2.resize(anchorimg,(224,224))
        self.positive=[l[0] for l in self.dataset if l[1]==label and l[0]!=sample]
        self.negative=[l[0] for l in self.dataset if l[1]!=label]
        # print(len(self.dataset),len(self.positive),len(self.negative))
        positive_index=random.choice(self.positive)
        negative_index=random.choice(self.negative)
        positiveimg=cv2.imread(os.path.join(self.datapath, positive_index))
        negativeimg=cv2.imread(os.path.join(self.datapath, negative_index))
        positiveimg = cv2.resize(positiveimg,(224,224))
        negativeimg = cv2.resize(negativeimg,(224,224))
        if self.dataselect == 'train' and self.aug: 
            anchorimg = self.aug(image=anchorimg)['image']
            positiveimg = self.aug(image=positiveimg)['image']
            negativeimg = self.aug(image=negativeimg)['image']
        anchorimg = self.trans(anchorimg)
        positiveimg = self.trans(positiveimg)
        negativeimg = self.trans(negativeimg)
        return anchorimg,positiveimg,negativeimg,label
        # else:
        #     anchorimg = self.aug(image=anchorimg)['image']
        #     anchorimg = self.trans(anchorimg)
        #     return anchorimg,label

    def __len__(self):
        return len(self.dataset)

    def getlabels(self):
        return self.labels


class TripletDFDataset3(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.patchsize=32
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = trainset
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            print(len(testset))
            self.dataset = testset
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]  
        random.shuffle(self.dataset)
        self.labels=[l[1] for l in self.dataset]


    def __getitem__(self, item):
        sample,label,catelabel = self.dataset[item]
        # print(os.path.join(self.datapath, sample))
        self.labels.append(label)
        image_size=224
        anchorimg = cv2.imread(os.path.join(self.datapath, sample))
        # print(anchorimg.shape)
        anchorimg = cv2.resize(anchorimg,(image_size,image_size))
        self.positive=[l[0] for l in self.dataset if l[1]==label and l[0]!=sample]
        self.negative=[l[0] for l in self.dataset if l[1]!=label]
        # print(len(self.dataset),len(self.positive),len(self.negative))
        positive_index=random.choice(self.positive)
        negative_index=random.choice(self.negative)
        positiveimg=cv2.imread(os.path.join(self.datapath, positive_index))
        negativeimg=cv2.imread(os.path.join(self.datapath, negative_index))
        positiveimg = cv2.resize(positiveimg,(image_size,image_size))
        negativeimg = cv2.resize(negativeimg,(image_size,image_size))
        if label==0:
            maskimg=np.zeros((image_size,image_size))
            label=[1,0]
        else:
            maskpath=os.path.join(self.datapath, sample)
            maskpath=maskpath.replace('c23/train','mask')
            maskpath=maskpath.replace('c23/test','mask')
            maskpath=maskpath.replace('c40/train','mask')
            maskpath=maskpath.replace('c40/test','mask')
            maskpath=maskpath.replace('raw/train','mask')
            maskpath=maskpath.replace('raw/test','mask')
            # print(os.path.join(self.datapath, sample),maskpath)
            # try:
            maskimg=cv2.imread(maskpath,0)
            maskimg = cv2.resize(maskimg,(image_size,image_size))
            maskimg=maskimg>20
            maskimg=np.array(maskimg,dtype=np.int32)
            label=[0,1]
            # nonzero=maskimg.nonzero()
            # startx,starty=nonzero[0][0],nonzero[1][0]
            # endx,endy=nonzero[0][-1],nonzero[1][-1]
            # print(maskpath,startx,starty,endx,endy)
            # except:
            #     print(maskpath)
        # print(maskpath)
        if self.dataselect == 'train': 
            anchorimg = self.aug(image=anchorimg)['image']
            positiveimg = self.aug(image=positiveimg)['image']
            negativeimg = self.aug(image=negativeimg)['image']
        anchorimg = self.trans(anchorimg)
        positiveimg = self.trans(positiveimg)
        negativeimg = self.trans(negativeimg)
        maskimg=transforms.ToTensor()(maskimg)
        maskimg=maskimg.float()
        # print(maskimg.shape)
        label=torch.tensor(label)
        # label=label.reshape((1))
        label=label.float()
        # maskimg2=1-maskimg
        # maskimg=torch.cat([maskimg,maskimg2],axis=0)

        return anchorimg,positiveimg,negativeimg,maskimg,label

    def __len__(self):
        return len(self.dataset)

    def getlabels(self):
        return self.labels

class TripletDFDataset3(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath,exceptdata=None):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.patchsize=32
        self.exceptdata=exceptdata
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = trainset
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            print(len(testset))
            self.dataset = testset
        self.dataset=[d for d in self.dataset if d[1] not in self.exceptdata] 
        random.shuffle(self.dataset)
        self.labels=[l[1] for l in self.dataset]


    def __getitem__(self, item):
        sample,label,_ = self.dataset[item]
        # print(os.path.join(self.datapath, sample))
        self.labels.append(label)
        image_size=224
        anchorimg = cv2.imread(os.path.join(self.datapath, sample))
        # print(anchorimg.shape)
        anchorimg = cv2.resize(anchorimg,(image_size,image_size))
        self.positive=[l[0] for l in self.dataset if l[1]==label and l[0]!=sample]
        self.negative=[l[0] for l in self.dataset if l[1]!=label]
        # print(len(self.dataset),len(self.positive),len(self.negative))
        positive_index=random.choice(self.positive)
        negative_index=random.choice(self.negative)
        positiveimg=cv2.imread(os.path.join(self.datapath, positive_index))
        negativeimg=cv2.imread(os.path.join(self.datapath, negative_index))
        positiveimg = cv2.resize(positiveimg,(image_size,image_size))
        negativeimg = cv2.resize(negativeimg,(image_size,image_size))
        if label==0:
            maskimg=np.zeros((image_size,image_size))
            label=[1,0]
        else:
            maskpath=os.path.join(self.datapath, sample.replace('newswap2/train','newmask'))
            maskpath=maskpath.replace('newswap2/test','newmask')
            # print(os.path.join(self.datapath, sample),maskpath)
            # try:
            # print(maskpath)
            maskimg=cv2.imread(maskpath,0)
            maskimg = cv2.resize(maskimg,(image_size,image_size))
            maskimg=maskimg>20
            maskimg=np.array(maskimg,dtype=np.int32)
            label=[0,1]
            # nonzero=maskimg.nonzero()
            # startx,starty=nonzero[0][0],nonzero[1][0]
            # endx,endy=nonzero[0][-1],nonzero[1][-1]
            # print(maskpath,startx,starty,endx,endy)
            # except:
            #     print(maskpath)
        # print(maskpath)
        if self.dataselect == 'train': 
            anchorimg = self.aug(image=anchorimg)['image']
            positiveimg = self.aug(image=positiveimg)['image']
            negativeimg = self.aug(image=negativeimg)['image']
        anchorimg = self.trans(anchorimg)
        positiveimg = self.trans(positiveimg)
        negativeimg = self.trans(negativeimg)
        maskimg=transforms.ToTensor()(maskimg)
        maskimg=maskimg.float()
        # print(maskimg.shape)
        label=torch.tensor(label)
        # label=label.reshape((1))
        label=label.float()
        # maskimg2=1-maskimg
        # maskimg=torch.cat([maskimg,maskimg2],axis=0)

        return anchorimg,positiveimg,negativeimg,maskimg,label

    def __len__(self):
        return len(self.dataset)

    def getlabels(self):
        return self.labels


class TripletDFDataset4(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.patchsize=32
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = trainset
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            print(len(testset))
            self.dataset = testset
        # self.dataset=self.dataset[:100]+self.dataset[-100:]    
        random.shuffle(self.dataset)
        self.labels=[l[1] for l in self.dataset]

    def cutimg(self,img,startx,starty,endx,endy):
        imgs=[]
        h,w,_=img.shape
        for i in range(2):
            randomx=random.randint(startx,endx-self.patchsize)
            randomy=random.randint(starty,endy-self.patchsize)
            timg=img[randomx:randomx+self.patchsize,randomy:randomy+self.patchsize,:]
            imgs.append(self.trans(timg))
        count=0
        while len(imgs)==2 and count<3:
            count+=1
            if startx-self.patchsize>0 and w-self.patchsize>0:
                randomx=random.randint(0,startx-self.patchsize)
                randomy=random.randint(0,w-self.patchsize)
                timg=img[randomx:randomx+self.patchsize,randomy:randomy+self.patchsize,:]
                imgs.append(self.trans(timg))
                return imgs
        count=0
        while len(imgs)==2 and count<3:
            count+=1
            if h-self.patchsize>0 and w-self.patchsize>0:
                randomx=random.randint(0,h-self.patchsize)
                randomy=random.randint(endy,w-self.patchsize)
                timg=img[randomx:randomx+self.patchsize,randomy:randomy+self.patchsize,:]
                imgs.append(self.trans(timg))
                return imgs
        count=0
        while len(imgs)==2 and count<3:
            count+=1
            if startx-self.patchsize>0 and starty-self.patchsize>0:
                randomx=random.randint(0,h-self.patchsize)
                randomy=random.randint(0,starty-self.patchsize)
                timg=img[randomx:randomx+self.patchsize,randomy:randomy+self.patchsize,:]
                imgs.append(self.trans(timg))
                return imgs
        count=0
        while len(imgs)==2 and count<3:
            count+=1
            if h-self.patchsize>0 and w-self.patchsize>0:
                randomx=random.randint(endx,h-self.patchsize)
                randomy=random.randint(0,w-self.patchsize)
                timg=img[randomx:randomx+self.patchsize,randomy:randomy+self.patchsize,:]
                imgs.append(self.trans(timg))
                return imgs

    def cutrealimg(self,img):
        imgs=[]
        h,w,_=img.shape
        for i in range(3):
            randomx=random.randint(0,h-self.patchsize)
            randomy=random.randint(0,w-self.patchsize)
            timg=img[randomx:randomx+self.patchsize,randomy:randomy+self.patchsize,:]
            imgs.append(self.trans(timg))
        return imgs

    def __getitem__(self, item):
        sample,label = self.dataset[item]
        # print(os.path.join(self.datapath, sample))
        self.labels.append(label)
        
        anchorimg = cv2.imread(os.path.join(self.datapath, sample))
        # print(anchorimg.shape)
        anchorimg = cv2.resize(anchorimg,(299,299))
        if label==0:
            maskimg=np.zeros((299,299))
            anchorimgs = self.cutrealimg(anchorimg)
        else:
            maskpath=os.path.join(self.datapath, sample)
            maskpath=maskpath.replace('c23/train','mask')
            maskpath=maskpath.replace('c23/test','mask')
            # print(maskpath)
            # try:
            maskimg=cv2.imread(maskpath,0)
            maskimg = cv2.resize(maskimg,(299,299))
            maskimg=maskimg>100
            maskimg=np.array(maskimg,dtype=np.int32)
            # print(maskpath,startx,starty,endx,endy)
            # except:
            #     print(maskpath)
        # print(maskpath)
        self.positive=[l[0] for l in self.dataset if l[1]==label and l[0]!=sample]
        self.negative=[l[0] for l in self.dataset if l[1]!=label]
        # print(len(self.dataset),len(self.positive),len(self.negative))
        positive_index=random.choice(self.positive)
        negative_index=random.choice(self.negative)
        positiveimg=cv2.imread(os.path.join(self.datapath, positive_index))
        negativeimg=cv2.imread(os.path.join(self.datapath, negative_index))
        positiveimg = cv2.resize(positiveimg,(299,299))
        negativeimg = cv2.resize(negativeimg,(299,299))
        
        if self.dataselect == 'train': 
            anchorimg = self.aug(image=anchorimg)['image']
            positiveimg = self.aug(image=positiveimg)['image']
            negativeimg = self.aug(image=negativeimg)['image']
        anchorimg = self.trans(anchorimg)
        positiveimg = self.trans(positiveimg)
        negativeimg = self.trans(negativeimg)
        maskimg=transforms.ToTensor()(maskimg)
        maskimg=maskimg.float()
        # print(anchorimgs.shape)
        return anchorimg,positiveimg,negativeimg,maskimg,label

    def __len__(self):
        return len(self.dataset)

    def getlabels(self):
        return self.labels


class AdaptiveDFDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath,exceptdata=None):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.patchsize=32
        self.exceptdata=exceptdata
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = trainset
            if self.exceptdata!=None:
                self.dataset=[l for l in self.dataset if l[2]!=self.exceptdata]  
            print('traindataset',len(self.dataset))
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']

            # testset = [d for d in testset if d[1]==1]
            
            self.dataset = testset
            if self.exceptdata!=None:
                # print(self.exceptdata)
                self.dataset=[l for l in self.dataset if l[2]==self.exceptdata] 
            print('testset',len(self.dataset)) 
        random.shuffle(self.dataset)
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]
        
        self.labels=[l[1] for l in self.dataset]
        self.catelabels=[l[2] for l in self.dataset]

    def __getitem__(self, item):
        sample,label,catelabel = self.dataset[item]
        if catelabel==0:
            catelabel=[1,0,0,0]
        elif catelabel==1:
            catelabel=[0,1,0,0]
        elif catelabel==2:
            catelabel=[0,0,1,0]
        elif catelabel==3:
            catelabel=[0,0,0,1]
        self.labels.append(label)
        image_size=224
        anchorimg = cv2.imread(os.path.join(self.datapath, sample))
        # print(anchorimg.shape)
        anchorimg = cv2.resize(anchorimg,(image_size,image_size))
        if label==0:
            maskimg=np.zeros((image_size,image_size))
            label=[1,0]
        else:
            maskpath=os.path.join(self.datapath, sample)
            maskpath=maskpath.replace('c23/train','mask')
            maskpath=maskpath.replace('c23/test','mask')
            maskpath=maskpath.replace('c40/train','mask')
            maskpath=maskpath.replace('c40/test','mask')
            maskpath=maskpath.replace('raw/train','mask')
            maskpath=maskpath.replace('raw/test','mask')
            # print(os.path.join(self.datapath, sample),maskpath)
            # try:
            maskimg=cv2.imread(maskpath,0)
            maskimg = cv2.resize(maskimg,(image_size,image_size))
            maskimg=maskimg>20
            maskimg=np.array(maskimg,dtype=np.int32)
            label=[0,1]
        if self.dataselect == 'train': 
            anchorimg = self.aug(image=anchorimg)['image']
        anchorimg = self.trans(anchorimg)
        maskimg=transforms.ToTensor()(maskimg)
        maskimg=maskimg.float()
        label=torch.tensor(label)
        label=label.float()
        catelabel=torch.tensor(catelabel)
        # catelabel = torch.zeros(1, 4).scatter_(1, catelabel, 1)
        catelabel=catelabel.float()
        return anchorimg,label,maskimg,catelabel

    def __len__(self):
        return len(self.dataset)

    def getlabels(self):
        return self.labels


class CelebDFAdaptiveDFDataset(Dataset):
    def __init__(self, datapath, dataselect, trans, augment, jsonpath):
        self.datapath = datapath
        self.trans = trans
        self.aug = augment
        self.num_classes = 2
        self.jsonpath = jsonpath
        self.dataselect = dataselect
        self.patchsize=32
        self.next_epoch()
        
    def next_epoch(self):
        with open(self.jsonpath, 'r') as f:
            data = json.load(f)
        if self.dataselect == 'train':
            trainset = data['train']
            self.dataset = trainset
        if self.dataselect == 'val':
            valset = data['val']
            print(len(valset))
            self.dataset = valset
        if self.dataselect == 'test':
            testset = data['test']
            print(len(testset))
            self.dataset = testset
        
        random.shuffle(self.dataset)
        # self.dataset=self.dataset[:100]  + self.dataset[-100:]  
        self.labels=[l[1] for l in self.dataset]

    def __getitem__(self, item):
        sample,label = self.dataset[item]
        self.labels.append(label)
        image_size=224
        anchorimg = cv2.imread(sample)
        # print(anchorimg.shape)
        anchorimg = cv2.resize(anchorimg,(image_size,image_size))
        if label==0:
            label=[1,0]
        else:
            label=[0,1]
        anchorimg = self.trans(anchorimg)
        label=torch.tensor(label)
        label=label.float()
        return anchorimg,label

    def __len__(self):
        return len(self.dataset)

    def getlabels(self):
        return self.labels