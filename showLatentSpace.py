# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:15:07 2019

@author: jprost
"""

import matplotlib.pyplot as plt
from AAEmodel import Encoder
import torch
from torchvision import transforms, datasets
import numpy as np
import pandas as pd

LATENT_DIM = 8
SIZE = 28

#load encoder
PATH_ENCODER = 'enc8_499.pt'
encoder = Encoder(SIZE**2,LATENT_DIM)
param = torch.load(PATH_ENCODER,map_location='cpu')
encoder.load_state_dict(param)
encoder.eval()

#%% load data
transform = transforms.Compose( [ transforms.ToTensor()])
                                  
dataset = datasets.MNIST("~/data/mnist", train=True, download=True,
                         transform=transform )
       
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)
n = len(dataloader)

#%%
#list of encoded vectors
Z = np.zeros((n,LATENT_DIM))
#list of labels
L = np.zeros(n)

#compute the encoded vectors of the data
for i,(im,lab) in enumerate(dataloader):
    v = im.view(-1,SIZE**2)
    z = encoder(v)
    Z[i] = z[0].detach().numpy()
    L[i] = lab.item()
label = pd.DataFrame(L,columns=['label'])
#%% 2D PCA 

from sklearn.decomposition import PCA
from matplotlib.cm import rainbow
import random
pca2d = PCA(n_components = 2)

pcDf= pca2d.fit_transform(Z)

dfPC2 = pd.DataFrame(data = pcDf
             , columns = ['pc1', 'pc2'])
             
finalDf = pd.concat([dfPC2, label], axis = 1)

fig,ax = plt.subplots()
colors = ['blue','orange','green','purple','red','cyan','pink','olive','brown','gray']
for i,group in enumerate(finalDf.groupby('label')):
    ax.plot(group[1]['pc1'],group[1]['pc2'],marker='o', linestyle='',linewidth=1, label=group[0],c=colors[i])
plt.legend()
plt.show()

#%%3d PCA 

pca3d = PCA(n_components = 3)

pcDf= pca3d.fit_transform(Z)

dfPC3 = pd.DataFrame(data = pcDf
             , columns = ['pc1', 'pc2','pc3'])
             
finalDf = pd.concat([dfPC3, label], axis = 1)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['blue','orange','green','purple','red','cyan','pink','olive','brown','gray']
for i,group in enumerate(finalDf.groupby('label')):
    ax.scatter(group[1]['pc1'],group[1]['pc2'],group[1]['pc3'],marker='o',label=group[0],c=colors[i])
plt.legend()
plt.show()