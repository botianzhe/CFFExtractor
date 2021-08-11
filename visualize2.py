# coding=utf-8
 
from sklearn.manifold import TSNE 
from pandas.core.frame import DataFrame
import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
# matplotlib.use('Agg')
#用TSNE进行数据降维并展示聚类结果

def vis(embeddings,labels):
    count=0
    for i in range(embeddings.shape[0]):
        t1=embeddings[i]
        count+=t1.shape[0]

    feature=np.zeros((count,320))
    label=np.zeros((count))
    # print(feature.shape,label.shape)
    index=0
    for i in range(embeddings.shape[0]):
        t1=embeddings[i]
        t2=labels[i]
        # print(t1.shape,t2.shape)
        feature[index:index+t1.shape[0]]=t1
        label[index:index+t1.shape[0]]=t2
        index+=t1.shape[0]

    print(feature.shape,label.shape)
    return feature,label

dfembeddings=np.load('npy/df_all+test_embeddings.npy',allow_pickle=True)

fsembeddings=np.load('npy/fs_all+test_embeddings.npy',allow_pickle=True)
f2fembeddings=np.load('npy/f2f_all+test_embeddings.npy',allow_pickle=True)
ntembeddings=np.load('npy/nt_all+test_embeddings.npy',allow_pickle=True)

dfembeddings2=np.load('npy/df_raw.npy',allow_pickle=True)

fsembeddings2=np.load('npy/fs_raw.npy',allow_pickle=True)
f2fembeddings2=np.load('npy/f2f_raw.npy',allow_pickle=True)
ntembeddings2=np.load('npy/nt_raw.npy',allow_pickle=True)



dflabels=np.ones(dfembeddings.shape[0])
fslabels=np.ones(fsembeddings.shape[0])*2
f2flabels=np.ones(f2fembeddings.shape[0])*3
ntlabels=np.ones(ntembeddings.shape[0])*4

print(dfembeddings.shape,dfembeddings2.shape)
dffeature,dflabel=vis(dfembeddings,dflabels)
fsfeature,fslabel=vis(fsembeddings,fslabels)
f2ffeature,f2flabel=vis(f2fembeddings,f2flabels)
ntfeature,ntlabel=vis(ntembeddings,ntlabels)

dffeature2,dflabel=vis(dfembeddings2,dflabels)
fsfeature2,fslabel=vis(fsembeddings2,fslabels)
f2ffeature2,f2flabel=vis(f2fembeddings2,f2flabels)
ntfeature2,ntlabel=vis(ntembeddings2,ntlabels)

feature=np.concatenate([dffeature,fsfeature,f2ffeature,ntfeature,dffeature2,fsfeature2,f2ffeature2,ntfeature2],axis=0)
label=np.concatenate([dflabel,fslabel,f2flabel,ntlabel,dflabel+4,fslabel+4,f2flabel+4,ntlabel+4],axis=0)
print(feature.shape)
pca=PCA(n_components=3)
t2d_vector=pca.fit_transform(feature)
# feature=feature[:10,:]
# label=label[:10]
# std=StandardScaler()
# feature=std.fit_transform(feature)
# tsne = TSNE(n_components=2, learning_rate=100)
# tsne.fit_transform(feature) #进行数据降维,并返回结果




# t2d_vector = tsne.embedding_
print(t2d_vector.shape,label.shape)
dffake=t2d_vector[label==1]
fsfake=t2d_vector[label==2]
f2ffake=t2d_vector[label==3]
ntfake=t2d_vector[label==4]

dffake2=t2d_vector[label==5]
fsfake2=t2d_vector[label==6]
f2ffake2=t2d_vector[label==7]
ntfake2=t2d_vector[label==8]

plt.clf()
# plt.scatter(dffake[:,0], dffake[:,1], c='b')
# plt.scatter(fsfake[:,0], fsfake[:,1], c='r')
# plt.scatter(f2ffake[:,0], f2ffake[:,1], c='g')
# plt.scatter(ntfake[:,0], ntfake[:,1], cmap='BuGn')
fig = plt.figure(figsize=(16,12))
# ax = fig.add_subplot(1, 1, 1, projection='3d')
c=0
# for X in [dffake,fsfake,f2ffake,ntfake]:
#     x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
#     X = (X - x_min) / (x_max - x_min)
#     for i in range(X.shape[0]):
#         ax.text(X[i, 0], X[i, 1], X[i,2],str(label[i]),
#                     color=plt.cm.Set1(c / 10.),
#                     fontdict={'weight': 'bold', 'size': 9})
#     c+=1
font={
    'family':'Times New Roman',
    'weight':'normal',
    'size':23
}

ax =  fig.add_subplot(1, 1, 1, projection='3d')
# featurename=['DeepFake','FaceSwap','Face2Face','NeuralTexture','DeepFake_adaptive','FaceSwap_adaptive','Face2Face_adaptive','NeuralTexture_adaptive']
# for X in [dffake,fsfake,f2ffake,ntfake,dffake2,fsfake2,f2ffake2,ntfake2]:
featurename=['DeepFake_adaptive','FaceSwap_adaptive','Face2Face_adaptive','NeuralTexture_adaptive']
for X in [dffake2,fsfake2,f2ffake2,ntfake2]:
    # x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    # X = (X - x_min) / (x_max - x_min)
    ax.scatter(X[:, 0], X[:, 1], X[:,2], label=featurename[c],color=plt.cm.Set1(c)) 
    
    plt.legend(fontsize=19,markerscale=2)
    c+=1
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')    
        

plt.show()
# tsne = TSNE(n_components=3, init='pca', random_state=0)
# X_tsne = tsne.fit_transform(feature)
# def plot_embedding_3d(X, title=None):
#     #坐标缩放到[0,1]区间
#     x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
#     X = (X - x_min) / (x_max - x_min)
#     #降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1, projection='3d')
#     for i in range(X.shape[0]):
#         ax.text(X[i, 0], X[i, 1], X[i,2],str(label[i]),
#                  color=plt.cm.Set1(label[i] / 10.),
#                  fontdict={'weight': 'bold', 'size': 9})
#     if title is not None:
#         plt.title(title)