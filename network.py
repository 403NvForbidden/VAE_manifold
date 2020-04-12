# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-02T12:15:22+11:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-04-10T17:36:02+10:00

'''
File containing VAE architecture, to reconstruct 4x128x128 single cell images,
while learning a 2D latent space, for visualization purpose.
'''

import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_

class VAE(nn.Module):
    def __init__(self, zdim):
        super(VAE, self).__init__()
        '''
        param :
            zdim (int) : dimension of the latent space
        '''
        # TODO: Modularize to any input depth (channels) and network depths

        self.zdim = zdim

        ##### Encoding layers #####
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(4, 4), padding=(1, 1), stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), padding=(1, 1), stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), padding=(1, 1), stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), padding=(1, 1), stride=2)
        self.bn4 = nn.BatchNorm2d(256)

        #mu
        self.fc11 = nn.Linear(8*8*256,512) #Feature extraction of 100 Features
        self.fc12 = nn.Linear(512,self.zdim)

        #logvar
        self.fc21 = nn.Linear(8*8*256,512) #Feature extraction of 100 Features
        self.fc22 = nn.Linear(512,self.zdim)

        ##### Decoding Layers #####
        self.fc3 = nn.Linear(self.zdim,512)
        self.fc4 = nn.Linear(512,8*8*256)

        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.bn6 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, padding=1, stride=2)
        self.bn7 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=4, kernel_size=4, padding=1, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_normal_(m.weight,gain=math.sqrt(2)) #Gain adapted for ReLU activation function
                m.bias.data.fill_(0.01)
            #elif isinstance(m, nn.BatchNorm2d):
            #    nn.init.constant_(m.weight, 1)
            #    nn.init.constant_(m.bias, 0)


    def encode(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x_fc = x.view(-1,8*8*256)

        mu_z = F.relu(self.fc11(x_fc))
        mu_z = self.fc12(mu_z)

        logvar_z = F.relu(self.fc21(x_fc))
        logvar_z = self.fc22(logvar_z)

        return mu_z, logvar_z


    def reparameterize(self, mu, logvar):
        if self.training: # if prediction, give the mean as a sample, which is the most likely
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) #random numbers from a normal distribution with mean 0 and variance 1
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self,z):
        '''
        Take as input a given sample of the latent space
        '''
        z = z.view(-1,self.zdim)
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))

        z_conv = z.view(-1,256,8,8)
        z_conv = F.relu(self.bn5(self.deconv1(z_conv)))
        z_conv = F.relu(self.bn6(self.deconv2(z_conv)))
        z_conv = F.relu(self.bn7(self.deconv3(z_conv)))
        x_mu = torch.sigmoid(self.deconv4(z_conv)) # TODO: CHECK if make sens to force 0-1, because output is an image

        #no x_logvar because it is assumed as a gaussian with idendity covariance matrix
        return x_mu

    def forward(self, x):
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z,logvar_z)
        x_rec = self.decode(z)

        return x_rec, mu_z, logvar_z


    def loss_function(self, x_recon, x, mu, logvar, beta):

        kl = - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl.sum() / x.size(0) #divide by batch size

        recon = 0.5 * (x_recon - x).pow(2)
        recon_loss = recon.sum() / x.size(0)

        loss = beta*kl_loss + recon_loss

        return loss, kl_loss, recon_loss
