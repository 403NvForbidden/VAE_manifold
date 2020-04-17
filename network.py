# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-02T12:15:22+11:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-04-14T10:59:34+10:00

'''
File containing VAE architecture, to reconstruct 4x128x128 single cell images,
while learning a 2D latent space, for visualization purpose.
'''

import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_


class Conv(nn.Module):
    '''Downsampling block'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv,self).__init__()

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
        super(ConvTranspose,self).__init__()

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
        super(ConvUpsampling,self).__init__()

        self.scale_factor = kernel_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        return self.conv(x)



class VAE(nn.Module):
    def __init__(self, zdim, channels=4, base=16, loss='MSE', layer_count=2, input_size=256):
        super(VAE, self).__init__()
        '''
        param :
            zdim (int) : dimension of the latent space
            channels (int) : number of channels of input data
            layer_count (int) : Depth of the network. Number of Conv - Block
            base (int) : First channel increase. Each bloc have *= 2 increase
            loss (string) : 'BCE' or 'MSE' loss will be used
            input_size (int) : shape of input data (optimized for 256x256)
        '''

        self.zdim = zdim
        self.channels = channels
        self.layer_count = layer_count
        self.base = base
        self.loss = loss
        self.input_size = input_size

        ###########################
        ##### Encoding layers #####
        ###########################
        out_channels = self.base
        encoding_modules = []
        encoding_modules.append(Conv(self.channels, self.base, 3, stride=2, padding=1)) #Downsampled two times - 128x128
        for i in range(self.layer_count):
            encoding_modules.append(Conv(out_channels,2*out_channels,3,padding=1)) # Keep same size
            encoding_modules.append(Conv(2*out_channels,2*out_channels,3,stride=2,padding=1)) #Downsampled two times
            encoding_modules.append(Conv(2*out_channels,2*out_channels,3,padding=1))
            encoding_modules.append(Conv(2*out_channels,2*out_channels,3,stride=2,padding=1))
            out_channels *= 2

        #fully connected layer (coded as convolution)
        dividing_factor = 2**((2*self.layer_count)+1)
        print(f'Given the depth you chose (layer_count = {self.layer_count}), input image are downsampled by half {(2*self.layer_count)+1} times')
        print(f'Before being fed to MLP, the shape is of {self.input_size/dividing_factor}')
        assert (self.input_size/dividing_factor)>=2, f'The network is to deep for the given input shape'
        encoding_modules.append(nn.Conv2d(out_channels,64*self.base,kernel_size=int(self.input_size/dividing_factor)))
        #Output a MLP of size 64*base  (1024 is base = 16)
        encoding_modules.append(nn.LeakyReLU())

        self.encoder = nn.Sequential(*encoding_modules)

        # MLP to mu and logvar
        self.encoder_mu = nn.Conv2d(64*base, self.zdim, 1) #Encode mu in desired latent dimension
        self.encoder_logvar = nn.Conv2d(64*base, self.zdim, 1) #Encode logvar in desired latent dimension


        ###########################
        ##### Decoding layers #####
        ###########################
        decoding_modules = []

        #MLP back to conv
        decoding_modules.append(nn.Conv2d(self.zdim,64*base,1))
        decoding_modules.append(ConvTranspose(64*base,out_channels,8))
        for i in range(self.layer_count):
            if i == 0:
                first_chan = out_channels
            else:
                first_chan = int(2*out_channels)
            decoding_modules.append(Conv(first_chan,out_channels,3,padding=1)) # Keep same size
            decoding_modules.append(ConvUpsampling(out_channels,out_channels,4,stride=2,padding=1)) # Upsampled two times
            decoding_modules.append(Conv(out_channels,out_channels,3,padding=1)) # Keep same size
            decoding_modules.append(ConvUpsampling(out_channels,out_channels,4,stride=2,padding=1)) # Upsampled two times
            out_channels = int(out_channels/2)

        decoding_modules.append(Conv(2*base,base,3,padding=1))
        decoding_modules.append(ConvUpsampling(self.base,self.base,4,stride=2,padding=1))
        decoding_modules.append(nn.Conv2d(self.base,self.channels,3,padding=1))

        self.decoder = nn.Sequential(*decoding_modules)



        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_normal_(m.weight,gain=math.sqrt(2./(1+0.01))) #Gain adapted for LeakyReLU activation function
                m.bias.data.fill_(0.01)
            #elif isinstance(m, nn.BatchNorm2d):
            #    nn.init.constant_(m.weight, 1)
            #    nn.init.constant_(m.bias, 0)


    def encode(self, x):
        '''Take input as batch. Dim batchx4x256x256'''
        x = self.encoder(x)

        mu_z = self.encoder_mu(x)
        logvar_z = self.encoder_logvar(x)

        return mu_z, logvar_z

    def reparameterize(self, mu, logvar):
        if self.training: # if prediction, give the mean as a sample, which is the most likely
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) #random numbers from a normal distribution with mean 0 and variance 1
            return mu + eps*std
        else:
            return mu

    def decode(self, z):
        x_mu = torch.sigmoid(self.decoder(z))
        #no x_logvar because it is assumed as a gaussian with idendity covariance matrix
        return x_mu

    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu,logvar)
        x_recon = self.decode(z)

        return x_recon, mu, logvar


    def loss_function(self, x_recon, x, mu, logvar, beta):
        #print(f'LogVar is : {torch.max(logvar)}')
        kl = - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl.sum() / x.size(0) #divide by batch size

        if self.loss == 'MSE':
            recon = 0.5 * (x_recon - x).pow(2)
            recon_loss = recon.sum() / x.size(0)
        elif self.loss == 'BCE':
            recon_loss_func = nn.BCELoss()
            recon_loss = recon_loss_func(x_recon,x)

        loss = beta*kl_loss + recon_loss
        #print(f'KL loss is : {kl_loss} and recon loss is : {recon_loss}')
        return loss, kl_loss, recon_loss



            # def encode(self,x):
            #     x = F.relu(self.bn1(self.conv1(x)))
            #     x = F.relu(self.bn2(self.conv2(x)))
            #     x = F.relu(self.bn3(self.conv3(x)))
            #     x = F.relu(self.bn4(self.conv4(x)))
            #
            #     x_fc = x.view(-1,8*8*256)
            #
            #     mu_z = F.relu(self.fc11(x_fc))
            #     mu_z = self.fc12(mu_z)
            #
            #     logvar_z = F.relu(self.fc21(x_fc))
            #     logvar_z = self.fc22(logvar_z)
            #
            #     return mu_z, logvar_z


            # def reparameterize(self, mu, logvar):
            #     if self.training: # if prediction, give the mean as a sample, which is the most likely
            #         std = torch.exp(0.5 * logvar)
            #         eps = torch.randn_like(std) #random numbers from a normal distribution with mean 0 and variance 1
            #         return eps.mul(std).add_(mu)
            #     else:
            #         return mu
            #
            # def decode(self,z):
            #     '''
            #     Take as input a given sample of the latent space
            #     '''
            #     z = z.view(-1,self.zdim)
            #     z = F.relu(self.fc3(z))
            #     z = F.relu(self.fc4(z))
            #
            #     z_conv = z.view(-1,256,8,8)
            #     z_conv = F.relu(self.bn5(self.deconv1(z_conv)))
            #     z_conv = F.relu(self.bn6(self.deconv2(z_conv)))
            #     z_conv = F.relu(self.bn7(self.deconv3(z_conv)))
            #     x_mu = torch.sigmoid(self.deconv4(z_conv)) # TODO: CHECK if make sens to force 0-1, because output is an image
            #
            #     #no x_logvar because it is assumed as a gaussian with idendity covariance matrix
            #     return x_mu
            #
            # def forward(self, x):
            #     mu_z, logvar_z = self.encode(x)
            #     z = self.reparameterize(mu_z,logvar_z)
            #     x_rec = self.decode(z)
            #
            #     return x_rec, mu_z, logvar_z




                    # ##### Encoding layers #####
                    # self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(4, 4), padding=(1, 1), stride=2)
                    # self.bn1 = nn.BatchNorm2d(32)
                    # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), padding=(1, 1), stride=2)
                    # self.bn2 = nn.BatchNorm2d(64)
                    # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), padding=(1, 1), stride=2)
                    # self.bn3 = nn.BatchNorm2d(128)
                    # self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), padding=(1, 1), stride=2)
                    # self.bn4 = nn.BatchNorm2d(256)
                    #
                    # #mu
                    # self.fc11 = nn.Linear(8*8*256,512) #Feature extraction of 100 Features
                    # self.fc12 = nn.Linear(512,self.zdim)
                    #
                    # #logvar
                    # self.fc21 = nn.Linear(8*8*256,512) #Feature extraction of 100 Features
                    # self.fc22 = nn.Linear(512,self.zdim)
                    #
                    # ##### Decoding Layers #####
                    # self.fc3 = nn.Linear(self.zdim,512)
                    # self.fc4 = nn.Linear(512,8*8*256)
                    #
                    # self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, padding=1, stride=2)
                    # self.bn5 = nn.BatchNorm2d(128)
                    # self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
                    # self.bn6 = nn.BatchNorm2d(64)
                    # self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, padding=1, stride=2)
                    # self.bn7 = nn.BatchNorm2d(32)
                    # self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=4, kernel_size=4, padding=1, stride=2)
