# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:42:04 2019

@author: jprost
"""

import torch
from torch import nn, optim, distributions
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import torch.nn.functional as F
from AAEmodel import Encoder, Decoder, Discriminator

import matplotlib.pyplot as plt
#%%

EPOCH_MAX = 100
LATENT_DIM = 2
IMAGE_SIZE = 28
INPUT_DIM = IMAGE_SIZE**2
BATCH_SIZE = 100

#%% load data
transform = transforms.Compose( [ transforms.ToTensor()])
                                  
dataset = datasets.MNIST("~/data/mnist", train=True, download=True,
                         transform=transform )
       
dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

#%% 


if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

def real_target(dim):
    return torch.ones(dim).to(device)
    
def fake_target(dim):
    return torch.zeros(dim).to(device)
    
#%%
def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

encoder = Encoder(INPUT_DIM,LATENT_DIM).to(device)
decoder = Decoder(LATENT_DIM, INPUT_DIM).to(device)
discriminator = Discriminator(LATENT_DIM).to(device)   

#optimizers
#reconstruction
optim_REC = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()),lr = 0.001)
#generator (=encoder)
optim_GEN = torch.optim.SGD(encoder.parameters(),lr = 0.01)
#discriminator
optim_DISC = torch.optim.SGD(discriminator.parameters(),lr = 0.01)

#prior latent distribution
mu = torch.zeros(LATENT_DIM)
cov = 5*torch.eye(LATENT_DIM)
prior = torch.distributions.MultivariateNormal(mu,cov)

#draw sample from the prior to generate images
Z_sample = prior.sample(torch.Size([12])).to(device)

#reconstruction loss
rec_loss = nn.MSELoss()
#discriminator loss
disc_loss = nn.BCELoss()

def showImageGrid(ims):
    fig = plt.figure(figsize=(9,3))
    ax = [fig.add_subplot(2,6,i+1) for i in range(12)]
    for k,a in enumerate(ax):
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_aspect('equal')

        im = ims[k,0].numpy()
        a.imshow(im,cmap='gray')
    plt.show()
#%%

for epoch in range(200):
    S_rec_loss,S_disc_loss,S_reg_loss = 0,0,0
    for i,(imgs,_) in enumerate(dataloader):
        #transform 2D image to 1D tensor (batch,vector dim)
        vect_ims = imgs.view(-1,INPUT_DIM).to(device)
        
        #train auto encoder (reconstruction phase)
        encoder.zero_grad()
        decoder.zero_grad()
        Z = encoder(vect_ims)
        X_hat = decoder(Z)
        #reconstruction loss
        r_loss = rec_loss(X_hat,vect_ims)
        r_loss.backward()
        optim_REC.step()
            
        #train discriminator 
        discriminator.zero_grad()
        #generate encoding from real image (= fake sample of prior latent)
        Z_fake = encoder(vect_ims).detach()
        D_fake = discriminator(Z_fake)[:,0]
        loss_fake = disc_loss(D_fake,fake_target(D_fake.size(0)))
        loss_fake.backward()
        err_fake = loss_fake.item()
        #draw "real" sample from prior
        Z_real = prior.sample(torch.Size([BATCH_SIZE])).to(device)
        D_real = discriminator(Z_real)[:,0]
        loss_real = disc_loss(D_real,real_target(D_real.size(0)))
        loss_real.backward()
        err_real = loss_real.item()
        optim_DISC.step()
        err_disc = err_real + err_fake
        
        #train encoder to mistake the discriminator (ie encoded z close to the prior)
        encoder.zero_grad()
        Z_fake = encoder(vect_ims)
        D_fake =  discriminator(Z_fake)[:,0]
        #aim to maximise the discriminator loss 
        reg_loss =  disc_loss(D_fake,real_target(D_real.size(0)))
        reg_loss.backward()
        optim_GEN.step()
        
        #mean losses
        S_rec_loss += r_loss.item()
        S_disc_loss += err_disc
        S_reg_loss += reg_loss.item()
        #display result
        if (i+1)%100 == 0:
            print('epoch {}, it = {}\n reconstrution {}, discriminator {}, regularization {}'.
                    format(epoch,i,S_rec_loss/100,S_disc_loss/100,S_reg_loss/100))
            S_rec_loss,S_disc_loss,S_reg_loss = 0,0,0
    images_vect = decoder(Z_sample).cpu().detach()
    imgs = vectors_to_images(images_vect)
    showImageGrid(imgs)

torch.save(encoder.state_dict(),'enc{}_{}.pt'.format(LATENT_DIM,epoch))
torch.save(decoder.state_dict(),'dec{}_{}.pt'.format(LATENT_DIM,epoch))
torch.save(discriminator.state_dict(),'disc{}_{}.pt'.format(LATENT_DIM,epoch))

print(epoch)
