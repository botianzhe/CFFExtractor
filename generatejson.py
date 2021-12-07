import os
import sys
import json
import shutil
import random

def generate_samples(reals,fakes):
    data=[]
    for r in reals:
        # print(r)
        rsamples=random.sample(reals,10)
        for rs in rsamples:
            fsamples=random.sample(fakes,10)
            for fs in fsamples:
                data.append([r,rs,fs,0])
    
    for f in fakes:
        fsamples=random.sample(fakes,10)
        for fs in fsamples:
            rsamples=random.sample(reals,10)
            for rs in rsamples:
                data.append([f,fs,rs,1])
    return data

# datapath='/Data/olddata_E/02YuPeipeng/deepfake/FF++/'
datapath1='/Data/olddata_D/ypp/mask/ff++/'
# target_datasets=['DF_Videos/','Real/']
for t in os.listdir(datapath1):
    if 'Deep' in t:
        target_datasets=[t,'Real']
        # target_datasets=['F2F_Videos/','Real']
        # target_datasets=['NT_Videos/','Real']
        for comp in ['raw','c23','c40']:
            traindata=[]
            valdata=[]
            testdata=[]
            for dataset in target_datasets:
                datapath=datapath1+dataset+'/'+comp+'/'
                print(datapath)
                for folder in ['train','test']:
                    folderpath=datapath+folder

                    print(folderpath)
                    for video in os.listdir(folderpath):
                        videopath=folderpath+'/'+video
                        # print(videopath)
                        for file in os.listdir(videopath):
                            filepath=videopath+'/'+file
                            recordpath= dataset+'/'+comp+'/videos/'+ folder+'/'+video+'/'+file
                            if folder=='train':
                                traindata.append(recordpath)
                            # if folder=='val':
                            #     valdata.append(recordpath)
                            if folder=='test':
                                testdata.append(recordpath)
            trainrealdata=[data for data in traindata if 'Real' in data]
            trainfakedata=[data for data in traindata if 'Real' not in data]
            testrealdata=[data for data in testdata if 'Real' in data]
            testfakedata=[data for data in testdata if 'Real' not in data]
            traindatas=generate_samples(trainrealdata,trainfakedata)
            testdatas=generate_samples(testrealdata,testfakedata)
            print(len(traindatas),len(testdatas))
            # print(len(trainfake))
            data={}
            data['train']=traindatas
            data['test']=testdatas

            with open('/Data/olddata_D/ypp/triplet/json/ff++_'+comp+'_'+target_datasets[0]+'.json', 'w') as json_file:
                json.dump(data,json_file)
            print(target_datasets[0],'finished')
