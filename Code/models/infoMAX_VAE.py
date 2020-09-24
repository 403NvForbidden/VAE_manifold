# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-04-29T14:42:50+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-31T10:19:46+10:00


"""File containing the architecture of InfoMax VAE, a VAE framework that
explicitly optimizes mutual information between observations and the latent
representations"""

import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from models.nn_modules import Conv, ConvUpsampling
import numpy as np


################################################
### Mutual information MLP estimator
################################################

class MLP_MI_estimator(nn.Module):
    """MLP, that take in input both input data and latent codes and output
    an unique value in R.
    This network defines the function t(x,z) that appears in the variational
    representation of a f-divergence. Finding t() that maximize this f-divergence,
    lead to a variation representation that is a tight bound estimator of mutual information.
    """

    def __init__(self, input_dim, zdim=3):
        super(MLP_MI_estimator, self).__init__()

        self.input_dim = input_dim
        self.zdim = zdim
        self.epochs = 0

        self.MLP_g = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
        )
        self.MLP_h = nn.Sequential(
            nn.Linear(zdim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )

    def forward(self, x, z):
        x = x.view(-1, self.input_dim)
        z = z.view(-1, self.zdim)
        x_g = self.MLP_g(x)  # Batchsize x 32
        y_h = self.MLP_h(z)  # Batchsize x 32
        scores = torch.matmul(y_h, torch.transpose(x_g, 0, 1))

        return scores  # Each element i,j is a scalar in R. f(xi,proj_j)


# Compute the Noise Constrastive Estimation (NCE) loss
def infoNCE_bound(scores):
    """Bound from Van Den Oord and al. (2018)"""
    nll = torch.mean(torch.diag(scores) - torch.logsumexp(scores, dim=1))
    k = scores.size()[0]
    mi = np.log(k) + nll

    return mi


################################################
### CNN-VAE architecture
################################################
class CNN_VAE(nn.Module):

    def __init__(self, zdim=3, input_channels=3, alpha=1, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2):
        """
        Modulate the complexity of the modulate with parameter 'base_enc' and 'base_dec'
        """
        super(CNN_VAE, self).__init__()
        self.zdim = zdim
        self.beta = beta
        self.alpha = alpha
        self.input_channels = input_channels
        self.base_dec = base_dec

        self.conv_enc = nn.Sequential(
            Conv(self.input_channels, base_enc, 4, stride=2, padding=1),  # stride 2, resolution is splitted by half
            Conv(base_enc, base_enc * 2, 4, stride=2, padding=1),  # 16x16
            Conv(base_enc * 2, base_enc * 4, 4, stride=2, padding=1),  # 8x8
            Conv(base_enc * 4, base_enc * 8, 4, stride=2, padding=1),  # 4x4
            Conv(base_enc * 8, base_enc * 16, 4, stride=2, padding=1),  # 2x2
        )
        self.linear_enc = nn.Sequential(
            nn.Linear(2 * 2 * base_enc * 16, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.mu_logvar_gen = nn.Linear(256, self.zdim * 2)

        self.linear_dec = nn.Sequential(
            nn.Linear(self.zdim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * 2 * base_dec * 16),
            nn.BatchNorm1d(2 * 2 * base_dec * 16),
            nn.ReLU()
        )

        self.conv_dec = nn.Sequential(
            ConvUpsampling(base_dec * 16, base_dec * 8, 4, stride=2, padding=1),  # 4
            ConvUpsampling(base_dec * 8, base_dec * 4, 4, stride=2, padding=1),  # 8
            ConvUpsampling(base_dec * 4, base_dec * 2, 4, stride=2, padding=1),  # 16
            ConvUpsampling(base_dec * 2, base_dec, 4, stride=2, padding=1),  # 32
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(base_dec, self.input_channels, 4, 2, 1),  # 192
            # nn.Sigmoid(), #Sigmoid compute directly in the loss (more stable)
        )

        self.stabilize_exp = nn.Hardtanh(min_val=-6., max_val=2.)  # linear between min and max

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_normal_(m.weight,
                               gain=math.sqrt(2. / (1 + 0.01)))  # Gain adapted for LeakyReLU activation function
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def reparameterize(self, mu, logvar):
        if self.training:  # if prediction, give the mean as a sample, which is the most likely
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)  # random numbers from a normal distribution with mean 0 and variance 1
            return mu + eps * std
        else:
            return mu

    def encode(self, x):
        batch_size = x.size(0)
        x = self.conv_enc(x)
        x = x.view((batch_size, -1))
        x = self.linear_enc(x)
        mu_logvar = self.mu_logvar_gen(x)
        mu_z, logvar_z = mu_logvar.view(-1, self.zdim, 2).unbind(-1)
        logvar_z = self.stabilize_exp(logvar_z)

        return mu_z, logvar_z

    def decode(self, z):
        batch_size = z.size(0)
        z = z.view((batch_size, -1))
        x = self.linear_dec(z)
        x = x.view((batch_size, self.base_dec * 16, 2, 2))
        x_recon = self.conv_dec(x)

        return x_recon

    def forward(self, x):

        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        x_recon = self.decode(z)
        return x_recon, mu_z, logvar_z, z.squeeze()

    def kl_divergence(self, mu, logvar):
        kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
        return kld

    def permute_dims(self, z):
        """Permutation in only a trick to be able to get sample from the marginal
        q(z) in a simple manner, to evaluate the variational representation of f-divergence
        """
        assert z.dim() == 2  # In the form of  batch x latentVariables
        B, _ = z.size()
        perm = torch.randperm(B).cuda()
        perm_z = z[perm]
        return perm_z


################################################################
######### InfoMAX VAE version for 128x128 inputs
################################################################

class CNN_128_VAE(nn.Module):

    def __init__(self, zdim=3, input_channels=4, alpha=1, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2):
        """
        Modulate the complexity of the decoder with parameter 'base_dec' and 'depth_factor_dec'
        """
        super(CNN_128_VAE, self).__init__()
        self.zdim = zdim
        self.beta = beta
        self.alpha = alpha
        self.input_channels = input_channels
        self.base_dec = base_dec

        self.conv_enc = nn.Sequential(
            Conv(self.input_channels, base_enc, 4, stride=2, padding=1),  # stride 2, resolution is splitted by half
            Conv(base_enc, base_enc * 2, 4, stride=2, padding=1),  # 32
            Conv(base_enc * 2, base_enc * 4, 4, stride=2, padding=1),  # 16
            Conv(base_enc * 4, base_enc * 8, 4, stride=2, padding=1),  # 8
            Conv(base_enc * 8, base_enc * 16, 4, stride=2, padding=1),  # 4
            Conv(base_enc * 16, base_enc * 16, 4, stride=2, padding=1),  # 2
        )
        self.linear_enc = nn.Sequential(
            nn.Linear(2 * 2 * base_enc * 16, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.mu_logvar_gen = nn.Linear(256, self.zdim * 2)

        self.linear_dec = nn.Sequential(
            nn.Linear(self.zdim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * 2 * base_dec * 16),
            nn.BatchNorm1d(2 * 2 * base_dec * 16),
            nn.ReLU()
        )

        self.conv_dec = nn.Sequential(
            ConvUpsampling(base_dec * 16, base_dec * 8, 4, stride=2, padding=1),  # 24
            ConvUpsampling(base_dec * 8, base_dec * 4, 4, stride=2, padding=1),  # 24
            ConvUpsampling(base_dec * 4, base_dec * 2, 4, stride=2, padding=1),  # 48
            ConvUpsampling(base_dec * 2, base_dec, 4, stride=2, padding=1),  # 96
            ConvUpsampling(base_dec, base_dec, 4, stride=2, padding=1),  # 96
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(base_dec, self.input_channels, 4, 2, 1),  # 192
            # nn.Sigmoid(), #Sigmoid compute directly in the loss (more stable)
        )

        self.stabilize_exp = nn.Hardtanh(min_val=-6., max_val=2.)  # linear between min and max
        # to constrain logvar in a reasonable range

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_normal_(m.weight,
                               gain=math.sqrt(2. / (1 + 0.01)))  # Gain adapted for LeakyReLU activation function
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def reparameterize(self, mu, logvar):
        if self.training:  # if prediction, give the mean as a sample, which is the most likely
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)  # random numbers from a normal distribution with mean 0 and variance 1
            return mu + eps * std
        else:
            return mu

    def encode(self, x):
        batch_size = x.size(0)
        x = self.conv_enc(x)
        x = x.view((batch_size, -1))
        x = self.linear_enc(x)
        mu_logvar = self.mu_logvar_gen(x)
        mu_z, logvar_z = mu_logvar.view(-1, self.zdim, 2).unbind(-1)
        logvar_z = self.stabilize_exp(logvar_z)

        return mu_z, logvar_z

    def decode(self, z):
        batch_size = z.size(0)
        z = z.view((batch_size, -1))
        x = self.linear_dec(z)
        x = x.view((batch_size, self.base_dec * 16, 2, 2))
        x_recon = self.conv_dec(x)

        return x_recon

    def forward(self, x):

        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        x_recon = self.decode(z)
        return x_recon, mu_z, logvar_z, z.squeeze()

    def kl_divergence(self, mu, logvar):
        kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
        return kld

    def permute_dims(self, z):
        """Permutation in only a trick to be able to get sample from the marginal
        q(z) in a simple manner, to evaluate the variational representation of f-divergence
        """
        assert z.dim() == 2  # In the form of  batch x latentVariables
        B, _ = z.size()
        perm = torch.randperm(B).cuda()
        perm_z = z[perm]
        return perm_z


class MLP_MI_128_estimator(nn.Module):
    """MLP, that take in input both input data and latent codes and output
    an unique value in R.
    This network defines the function t(x,z) that appears in the variational
    representation of a f-divergence. Finding t() that maximize this f-divergence,
    lead to a variation representation that is an exact estimator of mutual information.
    """

    def __init__(self, input_dim, zdim=3):
        super(MLP_MI_128_estimator, self).__init__()

        self.input_dim = input_dim
        self.zdim = zdim

        self.MLP_g = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
        )
        self.MLP_h = nn.Sequential(
            nn.Linear(zdim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )

    def forward(self, x, z):
        x = x.view(-1, self.input_dim)
        z = z.view(-1, self.zdim)
        x_g = self.MLP_g(x)  # Batchsize x 32
        y_h = self.MLP_h(z)  # Batchsize x 32
        scores = torch.matmul(y_h, torch.transpose(x_g, 0, 1))

        return scores
