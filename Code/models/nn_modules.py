# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-04-21T10:11:41+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-23T18:35:39+10:00

'''
File containing custom pytorch module, to be used as building blocks of VAE models.
c.f networks.py for the models.
'''

import torch
from torch import nn
from torch.nn import functional as F


###########################
###### standard VAE
###########################
class Decoder(nn.Module):
    '''Downsampling block'''

    def __init__(self, zdim: int = 3, input_channel=3, base_dec: int = 32, stride: int = 2, padding: int = 1):
        super(Decoder, self).__init__()

        ### p(z|y)
        # self.y_mu = nn.Linear(ydim, 2*zdim)
        # self.y_var = nn.Linear(ydim, zdim)
        self.base_dec = base_dec
        self.input_channels = input_channel

        self.linear_dec = nn.Sequential(
            Linear_block(zdim, 1024),
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
        x = x.view((batch_size, self.base_dec * 16, 2 ** ((self.input_channels - 1) // 2), 2 ** ((self.input_channels - 1) // 2)))
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
    '''Upsampling block, that interpolate instead of using ConvTranspose2d
    in order to mitigate the checkboard pattern effect'''

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


class ResBlock_identity(nn.Module):
    '''Basic Identity ResBlock. Input goes through :
    - Two Conv2D + BN + Activation
    - Shortcut identity mapping

    Both paths are added at the end. Keep same dimensionality'''

    def __init__(self, channels, kernel_size=(3, 3)):
        super(ConvUpsampling, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        # shortcut path
        identity = x
        # main path
        x = self.conv1(x)
        x = self.conv2(x)
        # add
        x += idendity
        out = self.activation(x)

        return out


class ResBlock_Conv(nn.Module):
    '''DownSampling ResBlock. Input goes through :
    - Two Conv2D + BN + Activation. First one double the depth and halve the resolution
    - Shortcut with one conv2D to match dimension

    Both paths are added at the end. Keep same dimensionality'''

    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvUpsampling, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1),  # Resolution / 2
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),  # Keep same dim
            nn.BatchNorm2d(out_channels),
        )
        self.convShortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1),  # Resolution / 2
            nn.BatchNorm2d(out_channels)
        )
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        # shortcut path
        shortcut = self.convShortcut(x)
        # main path
        x = self.conv1(x)
        x = self.conv2(x)
        # add
        x += shortcut
        out = self.activation(x)

        return out
