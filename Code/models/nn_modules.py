# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-04-21T10:11:41+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-23T18:35:39+10:00

'''
File containing custom pytorch module, to be used as building blocks of VAE models.
c.f networks_refactoring.py for the models.
'''
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


###########################
###### standard VAE
###########################
class Encoder_256(nn.Module):
    def __init__(self, out_size=256, input_channel=3, base_enc=32):
        super(Encoder_256, self).__init__()
        '''
        Down sampling
        usgae: 
        '''
        self.out_size = out_size
        self.base_enc = base_enc
        self.input_channels = input_channel

        # self.conv_enc = nn.Sequential(  # 256 x 256
        #     Conv(self.input_channels, base_enc, 4, stride=2, padding=1),  # 128 x 128
        #     Conv(base_enc, base_enc * 2, 4, stride=2, padding=1),  # 64x64
        #     Conv(base_enc * 2, base_enc * 4, 4, stride=2, padding=1),  # 32x32
        #     Conv(base_enc * 4, base_enc * 8, 4, stride=2, padding=1),  # 16x16
        #     Conv(base_enc * 8, base_enc * 16, 4, stride=2, padding=1),  # 8x8
        #     # Conv(base_enc * 16, base_enc * 16, 4, stride=2, padding=1),  # 4x4
        #     # Conv(base_enc * 16, base_enc * 16, 4, stride=2, padding=1),  # 2x2
        # )
        self.conv_enc = nn.Sequential(  # 256 x 256
            Conv(self.input_channels, base_enc, 4, stride=2, padding=1),  # 128 x 128
            Conv(base_enc, base_enc, 4, stride=2, padding=1),  # 64x64
            Conv(base_enc, base_enc, 4, stride=2, padding=1),  # 32x32
            Conv(base_enc, base_enc, 4, stride=2, padding=1),  # 16x16
            Conv(base_enc, base_enc, 4, stride=2, padding=1),  # 8x8x32
        )
        self.linear_enc = nn.Sequential(
            # Linear_block(pow(2, self.input_channels - 1) * base_enc * 8, 1024),
            Linear_block(pow(2, 3) * base_enc * 8, 1024),
            Linear_block(1024, 256),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_enc(x)
        x = x.view((batch_size, -1))
        x = self.linear_enc(x)
        return x


class Decoder_256(nn.Module):
    '''Downsampling block'''

    def __init__(self, zdim=3, input_channel=3, base_dec=32, stride=2, padding=1):
        super(Decoder_256, self).__init__()
        self.base_dec = base_dec
        self.input_channels = input_channel

        self.linear_dec = nn.Sequential(
            Linear_block(zdim, 256),
            Linear_block(256, 1024),
            Linear_block(1024, pow(2, 3) * base_dec * 8),
        )

        self.conv_dec = nn.Sequential(
            # ConvUpsampling(base_dec * 16, base_dec * 16, 4, stride=stride, padding=padding), # 2x2
            # ConvUpsampling(base_dec * 16, base_dec * 16, 4, stride=stride, padding=padding), # 4x4
            ConvUpsampling(base_dec, base_dec, 4, stride=stride, padding=padding),  # 8x8
            ConvUpsampling(base_dec, base_dec, 4, stride=stride, padding=padding),  # 16x16
            ConvUpsampling(base_dec, base_dec, 4, stride=stride, padding=padding),  # 32x32
            ConvUpsampling(base_dec, base_dec, 4, stride=stride, padding=padding),  # 64x64
            # for 4 channel
            # ConvUpsampling(base_dec, base_dec, 4, stride=2, padding=1),  # 96
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), # 128x128
            nn.Conv2d(in_channels=base_dec, out_channels=input_channel, kernel_size=4, stride=2, padding=1),  # 256 x 256
        )

        # self.conv_dec = nn.Sequential(
        #     # ConvUpsampling(base_dec * 16, base_dec * 16, 4, stride=stride, padding=padding), # 2x2
        #     # ConvUpsampling(base_dec * 16, base_dec * 16, 4, stride=stride, padding=padding), # 4x4
        #     ConvUpsampling(base_dec * 16, base_dec * 8, 4, stride=stride, padding=padding),  # 8x8
        #     ConvUpsampling(base_dec * 8, base_dec * 4, 4, stride=stride, padding=padding),  # 16x16
        #     ConvUpsampling(base_dec * 4, base_dec * 2, 4, stride=stride, padding=padding),  # 32x32
        #     ConvUpsampling(base_dec * 2, base_dec, 4, stride=stride, padding=padding),  # 64x64
        #     # for 4 channel
        #     # ConvUpsampling(base_dec, base_dec, 4, stride=2, padding=1),  # 96
        #     nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), # 128x128
        #     nn.Conv2d(in_channels=base_dec, out_channels=input_channel, kernel_size=4, stride=2, padding=1),  # 256 x 256
        # )

    def forward(self, z):
        batch_size = z.size(0)
        z = z.view((batch_size, -1))
        x = self.linear_dec(z)
        edge_size = 8#2 ** ((self.input_channels - 1)) #// 2)
        x = x.view((batch_size, self.base_dec, edge_size, edge_size))
        x_recon = self.conv_dec(x)
        return x_recon


class Encoder(nn.Module):
    def __init__(self, out_size=256, input_channel=3, base_enc=32):
        super(Encoder, self).__init__()
        '''
        Down sampling
        usgae: 
        '''
        self.out_size = out_size
        self.base_enc = base_enc
        self.input_channels = input_channel

        self.conv_enc = nn.Sequential(
            Conv(self.input_channels, base_enc, 4, stride=2, padding=1),  # stride 2, resolution is splitted by half
            Conv(base_enc, base_enc * 2, 4, stride=2, padding=1),  # 16x16
            Conv(base_enc * 2, base_enc * 4, 4, stride=2, padding=1),  # 8x8
            Conv(base_enc * 4, base_enc * 8, 4, stride=2, padding=1),  # 4x4
            Conv(base_enc * 8, base_enc * 16, 4, stride=2, padding=1),  # 2x2
        )
        self.linear_enc = nn.Sequential(
            Linear_block(pow(2, self.input_channels - 1) * base_enc * 16, 1024),
            Linear_block(1024, 256),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_enc(x)
        x = x.view((batch_size, -1))
        x = self.linear_enc(x)
        return x


class Decoder(nn.Module):
    '''Downsampling block'''

    def __init__(self, zdim=3, input_channel=3, base_dec=32, stride=2, padding=1):
        super(Decoder, self).__init__()

        ### p(z|y)
        # self.y_mu = nn.Linear(ydim, 2*zdim)
        # self.y_var = nn.Linear(ydim, zdim)
        self.base_dec = base_dec
        self.input_channels = input_channel

        self.linear_dec = nn.Sequential(
            Linear_block(zdim, 256),
            Linear_block(256, 1024),
            Linear_block(1024, pow(2, input_channel - 1) * base_dec * 16),
        )

        self.conv_dec = nn.Sequential(
            ConvUpsampling(base_dec * 16, base_dec * 8, 4, stride=stride, padding=padding),  # 4
            ConvUpsampling(base_dec * 8, base_dec * 4, 4, stride=stride, padding=padding),  # 8
            ConvUpsampling(base_dec * 4, base_dec * 2, 4, stride=stride, padding=padding),  # 16
            ConvUpsampling(base_dec * 2, base_dec, 4, stride=stride, padding=padding),  # 32
            # for 4 channel
            # ConvUpsampling(base_dec, base_dec, 4, stride=2, padding=1),  # 96
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=base_dec, out_channels=input_channel, kernel_size=4, stride=2, padding=1),  # 192
        )

    def forward(self, z):
        batch_size = z.size(0)
        ### p(z|y) cluster to dim
        # y_mu = self.y_mu(y)
        # y_var = F.softplus(self.y_var(y))

        ### p(x|z) reconstruction
        z = z.view((batch_size, -1))
        x = self.linear_dec(z)
        edge_size = 2 ** ((self.input_channels - 1) // 2)
        x = x.view((batch_size, self.base_dec * 16, edge_size, edge_size))
        x_recon = self.conv_dec(x)
        return x_recon


###########################
###### Standard CNN
###########################
class Conv(nn.Module):
    '''Downsampling block'''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ConvTranspose(nn.Module):
    '''Upsampling block'''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvTranspose, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ConvUpsampling(nn.Module):
    '''
    Upsampling block, that interpolate instead of using ConvTranspose2d
    in order to mitigate the checkboard pattern effect
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvUpsampling, self).__init__()

        self.scale_factor = kernel_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        return self.conv(x)


###########################
###### Standard Linear
###########################
class Linear_block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropRate: int = 0.):
        super(Linear_block, self).__init__()

        self.liner = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropRate)
        )

    def forward(self, x):
        return self.liner(x)


###########################
###### Conv ResBlock to Keep Mutual Info high
###########################
class Skip_Conv_down(nn.Module):
    '''
    Standard Conv2d to learn best downsampling, but add a short cut skip
    layer to keep information as rich as possible.
    The downsampling of the short cut is an analytical interpolation, not learnable.
    Reduce the resolution by half
    '''

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, factor=0.5, mode='trilinear',
                 LastLayer=False):
        super(Skip_Conv_down, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.BN = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU()

        self.inc = in_channels
        self.ouc = out_channels
        self.down_factor = factor
        self.mode = mode
        self.LastLayer = LastLayer

    def forward(self, x):
        # Go to volumetric data (5D) to be able to interpolate over channels
        x_skip_down = F.interpolate(x.unsqueeze(dim=0),
                                    scale_factor=(self.ouc / self.inc, self.down_factor, self.down_factor),
                                    mode=self.mode)
        x = self.conv(x)
        if not (self.LastLayer):
            x = self.BN(x)
            return self.activation(x + x_skip_down.squeeze(dim=0))
        else:
            return x + x_skip_down.squeeze(dim=0)


class Skip_DeConv_up(nn.Module):
    '''
    Increase the resolution by a factor two
    '''

    def __init__(self, in_channels, out_channels, LastLayer=False):
        super(Skip_DeConv_up, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.BN = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU()
        self.LastLayer = LastLayer
        self.inc = in_channels
        self.ouc = out_channels

    def forward(self, x):
        # Go to volumetric data (5D) to be able to interpolate over channels
        x_skip_up = F.interpolate(x.unsqueeze(dim=0), scale_factor=(self.ouc / self.inc, 2, 2), mode='trilinear')
        x = self.conv(F.interpolate(x, scale_factor=4, mode='bilinear'))
        if not (self.LastLayer):
            x = self.BN(x)  # Replace a transpose conv
            return self.activation(x + x_skip_up.squeeze(dim=0))
        else:
            return x + x_skip_up.squeeze(0)


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
        # flatten
        x = x.view(-1, self.input_dim)
        z = z.view(-1, self.zdim)
        x_g = self.MLP_g(x)  # Batchsize x 32
        y_h = self.MLP_h(z)  # Batchsize x 32
        scores = torch.matmul(y_h, torch.transpose(x_g, 0, 1))

        return self.infoNCE_bound(scores)  # Each element i,j is a scalar in R. f(xi,proj_j)

    # Compute the Noise Constrastive Estimation (NCE) loss infoNCE_bound
    def infoNCE_bound(self, scores):
        """Bound from Van Den Oord and al. (2018)"""
        nll = torch.mean(torch.diag(scores) - torch.logsumexp(scores, dim=1))
        k = scores.size()[0]
        mi = np.log(k) + nll
        return mi


def infoNCE_bound(self, scores):
    """Bound from Van Den Oord and al. (2018)"""
    nll = torch.mean(torch.diag(scores) - torch.logsumexp(scores, dim=1))
    k = scores.size()[0]
    mi = np.log(k) + nll
    return mi
