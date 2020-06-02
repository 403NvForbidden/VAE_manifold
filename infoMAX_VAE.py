# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-04-29T14:42:50+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-06-01T15:46:14+10:00


'''File containing the architecture of InfoMax VAE, a VAE framework that
explicitly optimizes mutual information between observations and the latent
representations'''

import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from nn_modules import Conv, ConvUpsampling


################################################
### Mutual information MLP estimator
################################################

class MLP_MI_estimator(nn.Module):
    '''MLP, that take in input both input data and latent codes and output
    an unique value in R.
    This network defines the function t(x,z) that appears in the variational
    representation of a f-divergence. Finding t() that maximize this f-divergence,
    lead to a variation representation that is an exact estimator of mutual information.
    '''
    def __init__(self, zdim=3):
        super(MLP_MI_estimator, self).__init__()

        self.zdim = zdim
        ## TODO: This is for a 64x64 3 channels images data. Modularize it
        self.MLP = nn.Sequential(
            nn.Linear(64*64*3 + zdim, 2000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 100),
            nn.LeakyReLU(0.2, True),
            nn.Linear(100, 10),
            nn.LeakyReLU(0.2, True),
            nn.Linear(10, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_normal_(m.weight,gain=math.sqrt(2./(1+0.2))) #Gain adapted for LeakyReLU activation function
                m.bias.data.fill_(0.01)

    def forward(self, x, z):
        x = x.view(-1,64*64*3)
        x = torch.cat((x,z),1) #Simple concatenation of x and z
        value = self.MLP(x).squeeze()

        return value #Output is a scalar in R


################################################
### CNN-VAE architecture
################################################


class CNN_VAE(nn.Module):

    def __init__(self, zdim=3, alpha=1, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2):
        '''
        Modulate the complexity of the decoder with parameter 'base_dec' and 'depth_factor_dec'
        If depth_factor_dec = 2, and base are default value, encoder and decoder are symetric
        '''
        super(CNN_VAE, self).__init__()
        self.zdim = zdim
        self.beta = beta
        self.alpha = alpha

        # TODO: Here the VAE is fullz conv. for an input 64*64
        # Adapt that to any input size, and check if equivalent or not
        # to having reshape and dense layer at the end.

        self.encoder = nn.Sequential(
            Conv(3,20,3,stride=1,padding=1), # REMOVE
            Conv(20,base_enc,4,stride=2,padding=1), #stride 2, resolution is splitted by half
            Conv(base_enc,base_enc,3,stride=1,padding=1), # REMOVE
            Conv(base_enc,base_enc*2,4,stride=2,padding=1),
            Conv(base_enc*2,base_enc*2,3,stride=1,padding=1), # REMOVE
            Conv(base_enc*2,base_enc*4,4,stride=2,padding=1), #8x8
            Conv(base_enc*4,base_enc*8,4,stride=2,padding=1),
            Conv(base_enc*8,base_enc*16,4,stride=2,padding=1),  #2x2
            nn.Conv2d(base_enc*16, 2*zdim, 2, 1),#MLP  #1x1 -> encoded in channels

        )
        self.decoder = nn.Sequential(
            nn.Conv2d(zdim, int(base_dec*(depth_factor_dec**4)), 1), #MLP
            nn.LeakyReLU(),
            ConvUpsampling(int(base_dec*(depth_factor_dec**4)),int(base_dec*(depth_factor_dec**3)),4,stride=2,padding=1),
            ConvUpsampling(int(base_dec*(depth_factor_dec**3)),int(base_dec*(depth_factor_dec**2)),4,stride=2,padding=1),
            ConvUpsampling(int(base_dec*(depth_factor_dec**2)),int(base_dec*(depth_factor_dec**1)),4,stride=2,padding=1),
            ConvUpsampling(int(base_dec*(depth_factor_dec**1)),int(base_dec*(depth_factor_dec**0)),4,stride=2,padding=1),
            ConvUpsampling(int(base_dec*(depth_factor_dec**0)),int(base_dec*(depth_factor_dec**0)),4,stride=2,padding=1),

            nn.Upsample(scale_factor=4,mode='bilinear'),
            nn.Conv2d(int(base_dec*(depth_factor_dec**0)), 3, 4, 2, 1),
            #nn.Sigmoid(), #Sigmoid compute directly in the loss (more stable)
        )

        self.stabilize_exp = nn.Hardtanh(min_val=-6.,max_val=2.) #linear between min and max


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_normal_(m.weight,gain=math.sqrt(2./(1+0.01))) #Gain adapted for LeakyReLU activation function
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def reparameterize(self, mu, logvar):
        if self.training: # if prediction, give the mean as a sample, which is the most likely
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) #random numbers from a normal distribution with mean 0 and variance 1
            return mu + eps*std
        else:
            return mu

    def encode(self, x):
        learnt_stats = self.encoder(x) #Mu encode in the first half channels
        mu_z = learnt_stats[:,self.zdim:]
        logvar_z = self.stabilize_exp(learnt_stats[:, :self.zdim])

        return mu_z, logvar_z

    def decode(self, z):
        x_recon = self.decoder(z)

        return x_recon

    def forward(self, x):

        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        x_recon = self.decode(z)
        return x_recon, mu_z, logvar_z, z.squeeze()

    # def recon_loss(self, x_recon, x):
    #     '''
    #     Mean over batchsize, but not over dimension, to account for the difference
    #     of dimensionality between X and Z
    #     '''
    #     n = x.size(0)
    #     loss = F.binary_cross_entropy(x_recon, x).div(n)
    #     return loss


    def kl_divergence(self, mu, logvar):
        kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
        return kld

    def permute_dims(self, z):
        '''Permutation in only a trick to be able to get sample from the marginal
        q(z) in a simple manner, to evaluate the variational representation of f-divergence
        '''
        assert z.dim() == 2 #In the form of  batch x latentVariables
        B, _ = z.size()
        perm = torch.randperm(B).cuda()
        perm_z = z[perm]
        return perm_z
