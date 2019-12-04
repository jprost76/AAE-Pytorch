# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:33:26 2019

@author: jprost
"""

import matplotlib.pyplot as plt
from AAEmodel import Encoder,Decoder
import torch
from torchvision import transforms, datasets
import numpy as np
import pandas as pd


LATENT_DIM = 8
SIZE = 28
STD = 5
#%% load decoder
PATH_DECODER = 'dec8_199.pt'
decoder = Decoder(LATENT_DIM,SIZE**2)
param = torch.load(PATH_DECODER,map_location='cpu')
decoder.load_state_dict(param)
decoder.eval()

#prior latent distribution
mu = torch.zeros(LATENT_DIM)
cov = std**2 *torch.eye(LATENT_DIM)
prior = torch.distributions.MultivariateNormal(mu,cov)

#%%draw sample from the prior to generate images
Z_sample = prior.sample(torch.Size([64]))

#%%
vects = decoder(Z_sample).detach()

#%% generate new image
ims = vects.view(vects.size(0),1,28,28)
rand[0,0] = torch.ones(28,28)
def showImageGrid(ims):
    fig = plt.figure(figsize=(12,12))
    ax = [fig.add_subplot(8,8,i+1) for i in range(64)]
    for k,a in enumerate(ax):
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_aspect('equal')

        im = ims[k,0].numpy()
        a.imshow(im,cmap='gray')
    plt.show()
    
showImageGrid(ims)
