# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:16:51 2019

@author: jprost
"""

import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Gaussian MLP encoder as described in Appendix 1 A of AAE Paper (determinist case)
    """
    def __init__(self,input_dim, latent_dim):
        super(Encoder,self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim,1000)
        self.fc2 = nn.Linear(1000,1000)
        self.mu = nn.Linear(1000,latent_dim)
        #initialize the hidden layers weights
        self.fc1.weight.data.normal_(0,0.01)
        self.fc2.weight.data.normal_(0,0.01)
        self.mu.weight.data.normal_(0,0.01)
        
    def forward(self,x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu_z = self.mu(h)
        return mu_z
        
class Decoder(nn.Module):
    """
    decoder as described in Appendix 1 A of AAE Paper
    """
    def __init__(self,latent_dim,output_dim):
        super(Decoder,self).__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim,1000)
        self.fc2 = nn.Linear(1000,1000)
        self.fc3 = nn.Linear(1000,output_dim)
        #initialize the hidden layers weights
        self.fc1.weight.data.normal_(0,0.01)
        self.fc2.weight.data.normal_(0,0.01)
        self.fc3.weight.data.normal_(0,0.01)
           
    def forward(self,x):
        z = F.relu(self.fc1(x))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = torch.sigmoid(z)
        return z     
        
class Discriminator(nn.Module):
    """
    discriminator as described in Appendix 1 A of AAE Paper
    """
    def __init__(self,latent_dim):
        super(Discriminator,self).__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim,1000)
        self.fc2 = nn.Linear(1000,1000)
        self.fc3 = nn.Linear(1000,1)
        
    def forward(self,z):
        y = F.relu(self.fc1(z))
        y = F.relu(self.fc2(y))
        p = torch.sigmoid(self.fc3(y))
        return p
        
        
        
if __name__=="__main__":
    di = 28**2
    dz = 8
    enc = Encoder(di,dz)
    x = torch.randn(2,di)
    z = enc(x)
    print(z.size())
    dec = Decoder(dz,di)
    xh = dec(z)
    print(xh.size())
    D = Discriminator(dz)
    p = D(z)
    print(p.size())
        
