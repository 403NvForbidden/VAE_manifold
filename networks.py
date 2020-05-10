# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-02T12:15:22+11:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-05-07T12:26:25+10:00

'''
File containing different VAE architectures, to reconstruct 4xHxW single cell images,
while learning a 2D or 3D latent space, for visualization purpose and downstream data interrogation.
'''

import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from nn_modules import Conv, ConvTranspose, ConvUpsampling, Skip_Conv_down, Skip_DeConv_up



class VAE(nn.Module):
    def __init__(self, zdim=3, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2, loss='BCE'):
        super(VAE, self).__init__()
        '''
        param :
            zdim (int) : dimension of the latent space

            Modulate the complexity of the decoder with parameter 'base_dec' and 'depth_factor_dec'
            If depth_factor_dec = 2, and base are default value, encoder and decoder are symetric

        '''

        self.zdim = zdim
        self.beta = beta
        self.loss = loss


        ###########################
        ##### Encoding layers #####
        ###########################
        self.encoder = nn.Sequential(
            Conv(4,base_enc,4,stride=2,padding=1), #stride 2, resolution is splitted by half
            Conv(base_enc,base_enc*2,4,stride=2,padding=1),
            Conv(base_enc*2,base_enc*4,4,stride=2,padding=1), #8x8
            Conv(base_enc*4,base_enc*4,4,stride=2,padding=1),
            Conv(base_enc*4,base_enc*8,4,stride=2,padding=1),  #2x2
            nn.Conv2d(base_enc*8, 2*zdim, 2, 1),#MLP  #1x1 -> encoded in channels
        )

        ###########################
        ##### Decoding layers #####
        ###########################
        self.decoder = nn.Sequential(
            nn.Conv2d(zdim, int(base_dec*(depth_factor_dec**3)), 1), #MLP
            nn.LeakyReLU(),
            ConvUpsampling(int(base_dec*(depth_factor_dec**3)),int(base_dec*(depth_factor_dec**2)),4,stride=2,padding=1),
            ConvUpsampling(int(base_dec*(depth_factor_dec**2)),int(base_dec*(depth_factor_dec**2)),4,stride=2,padding=1),
            ConvUpsampling(int(base_dec*(depth_factor_dec**2)),int(base_dec*(depth_factor_dec**1)),4,stride=2,padding=1),
            ConvUpsampling(int(base_dec*(depth_factor_dec**1)),int(base_dec*(depth_factor_dec**0)),4,stride=2,padding=1),
            ConvUpsampling(int(base_dec*(depth_factor_dec**0)),int(base_dec*(depth_factor_dec**0)),4,stride=2,padding=1),

            nn.Upsample(scale_factor=4,mode='bilinear'),
            nn.Conv2d(int(base_dec*(depth_factor_dec**0)), 4, 4, 2, 1),
            #nn.Sigmoid(), #Sigmoid compute directly in the loss (more stable)
        )

        self.stabilize_exp = nn.Hardtanh(min_val=-6.,max_val=2.) #linear between min and max
        #to constrain logvar in a reasonable range


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

    def kl_divergence(self, mu, logvar):
        kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
        return kld



class Skip_VAE(nn.Module):

    def __init__(self, zdim, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2):
        '''
        Modulate the complexity of the decoder with parameter 'base_dec' and 'depth_factor_dec'
        If depth_factor_dec = 2, and base are default value, encoder and decoder are symetric
        '''
        super(Skip_VAE,self).__init__()
        self.zdim = zdim
        self.beta = beta

        self.encoder = nn.Sequential(
            Skip_Conv_down(4,base_enc), # resolution is splitted by half
            Skip_Conv_down(base_enc,base_enc*2,kernel_size=4, stride=2, padding=1),
            Skip_Conv_down(base_enc*2,base_enc*4,kernel_size=4, stride=2, padding=1),
            Skip_Conv_down(base_enc*4,base_enc*4,kernel_size=4, stride=2, padding=1),
            Skip_Conv_down(base_enc*4,base_enc*8,kernel_size=4, stride=2, padding=1), #HxW is 2x2
            Skip_Conv_down(base_enc*8,2*zdim,kernel_size=2,stride=1,padding=0,LastLayer=True) #Equivalent to a MLP
        )

        self.decoder = nn.Sequential(
            Skip_Conv_down(zdim,int(base_dec*(depth_factor_dec**3)),kernel_size=1,stride=1,padding=0,factor=1,mode='nearest'), #equivalent to a MLP
            Skip_DeConv_up(int(base_dec*(depth_factor_dec**3)),int(base_dec*(depth_factor_dec**2))),
            Skip_DeConv_up(int(base_dec*(depth_factor_dec**2)),int(base_dec*(depth_factor_dec**2))),
            Skip_DeConv_up(int(base_dec*(depth_factor_dec**2)),int(base_dec*(depth_factor_dec**1))),
            Skip_DeConv_up(int(base_dec*(depth_factor_dec**1)),int(base_dec*(depth_factor_dec**0))),
            Skip_DeConv_up(int(base_dec*(depth_factor_dec**0)),int(base_dec*(depth_factor_dec**0))),
            Skip_DeConv_up(int(base_dec*(depth_factor_dec**0)), 4, LastLayer=True)
        )

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
        logvar_z = learnt_stats[:, :self.zdim]

        return mu_z, logvar_z

    def decode(self, z):
        x_recon = self.decoder(z)

        return x_recon

    def forward(self, x):

        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        x_recon = self.decode(z)
        return x_recon, mu_z, logvar_z, z.squeeze()

    def kl_divergence(self, mu, logvar):
        kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
        return kld




# class VAE(nn.Module):
#     def __init__(self, zdim=3, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2):
#         super(VAE, self).__init__()
#         '''
#         param :
#             zdim (int) : dimension of the latent space
#             channels (int) : number of channels of input data
#             layer_count (int) : Depth of the network. Number of Conv - Block
#             base (int) : First channel increase. Each bloc have *= 2 increase
#             loss (string) : 'BCE' or 'MSE' loss will be used
#             input_size (int) : shape of input data (optimized for 256x256)
#         '''
#
#         self.zdim = zdim
#         self.channels = channels
#         self.layer_count = layer_count
#         self.base = base
#         self.loss = loss
#         self.input_size = input_size
#         self.model_type = 'VAE_CNN_vanilla' #For efficient saving and loading
#
#         ###########################
#         ##### Encoding layers #####
#         ###########################
#         out_channels = self.base
#         encoding_modules = []
#         encoding_modules.append(Conv(self.channels, self.base, 3, stride=2, padding=1)) #Downsampled two times - 128x128
#         for i in range(self.layer_count):
#             encoding_modules.append(Conv(out_channels,2*out_channels,3,padding=1)) # Keep same size
#             encoding_modules.append(Conv(2*out_channels,2*out_channels,3,stride=2,padding=1)) #Downsampled two times
#             encoding_modules.append(Conv(2*out_channels,2*out_channels,3,padding=1))
#             encoding_modules.append(Conv(2*out_channels,2*out_channels,3,stride=2,padding=1))
#             out_channels *= 2
#
#         #fully connected layer (coded as convolution)
#         dividing_factor = 2**((2*self.layer_count)+1)
#         print(f'Given the depth you chose (layer_count = {self.layer_count}), input image are downsampled by half {(2*self.layer_count)+1} times')
#         print(f'Before being fed to MLP, the shape is of {self.input_size/dividing_factor}')
#         assert (self.input_size/dividing_factor)>=2, f'The network is to deep for the given input shape'
#         encoding_modules.append(nn.Conv2d(out_channels,64*self.base,kernel_size=int(self.input_size/dividing_factor)))
#         #Output a MLP of size 64*base  (1024 is base = 16)
#         encoding_modules.append(nn.LeakyReLU())
#
#         self.encoder = nn.Sequential(*encoding_modules)
#
#         # MLP to mu and logvar
#         self.encoder_mu = nn.Conv2d(64*base, self.zdim, 1) #Encode mu in desired latent dimension
#         self.encoder_sqrtvar = nn.Conv2d(64*base, self.zdim, 1) #Encode sqrtvar in desired latent dimension
#         #NOTE : We encode Sqrt(var) and not logvar as in SOTA VAE, to leverage instability in loss with KL annealing caused by the term exp(logvar)
#         self.softplus = nn.Softplus()
#
#         ###########################
#         ##### Decoding layers #####
#         ###########################
#         decoding_modules = []
#
#         #MLP back to conv
#         decoding_modules.append(nn.Conv2d(self.zdim,64*base,1))
#         decoding_modules.append(ConvTranspose(64*base,out_channels,int(self.input_size/dividing_factor)))
#         for i in range(self.layer_count):
#             if i == 0:
#                 first_chan = out_channels
#             else:
#                 first_chan = int(2*out_channels)
#             decoding_modules.append(Conv(first_chan,out_channels,3,padding=1)) # Keep same size
#             decoding_modules.append(ConvUpsampling(out_channels,out_channels,4,stride=2,padding=1)) # Upsampled two times
#             decoding_modules.append(Conv(out_channels,out_channels,3,padding=1)) # Keep same size
#             decoding_modules.append(ConvUpsampling(out_channels,out_channels,4,stride=2,padding=1)) # Upsampled two times
#             out_channels = int(out_channels/2)
#
#         decoding_modules.append(Conv(2*base,base,3,padding=1))
#         decoding_modules.append(ConvUpsampling(self.base,self.base,4,stride=2,padding=1))
#         decoding_modules.append(nn.Conv2d(self.base,self.channels,3,padding=1))
#
#         self.decoder = nn.Sequential(*decoding_modules)
#
#
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 xavier_normal_(m.weight,gain=math.sqrt(2./(1+0.01))) #Gain adapted for LeakyReLU activation function
#                 m.bias.data.fill_(0.01)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#
#     def encode(self, x):
#         '''Take input as batch. Dim batchx4x256x256'''
#         x = self.encoder(x)
#
#         mu_z = self.encoder_mu(x)
#         sqrtvar_z = self.softplus(self.encoder_sqrtvar(x))
#
#         return mu_z, sqrtvar_z
#
#     def reparameterize(self, mu, sqrtvar):
#         if self.training: # if prediction, give the mean as a sample, which is the most likely
#             #std = torch.exp(0.5 * logvar)
#             std = sqrtvar
#             #NOTE : We encode Sqrt(var) and not logvar as in SOTA VAE, to leverage instability in loss with KL annealing caused by the term exp(logvar)
#
#             eps = torch.randn_like(std) #random numbers from a normal distribution with mean 0 and variance 1
#             return mu + eps*std
#         else:
#             return mu
#
#     def decode(self, z):
#         x_mu = torch.sigmoid(self.decoder(z))
#         #no x_logvar because it is assumed as a gaussian with idendity covariance matrix
#         return x_mu
#
#     def forward(self,x):
#         mu, sqrtvar = self.encode(x)
#         z = self.reparameterize(mu,sqrtvar)
#         x_recon = self.decode(z)
#
#         return x_recon, mu, sqrtvar
#
#
#     def loss_function(self, x_recon, x, mu, sqrtvar, beta):
#         #print(f'LogVar is : {torch.max(logvar)}')
#         kl = - 0.5 * (1 + 2*sqrtvar.log() - mu.pow(2) - sqrtvar.pow(2))
#         #NOTE : We encode Sqrt(var) and not logvar as in SOTA VAE, to leverage instability in loss with KL annealing caused by the term exp(logvar)
#         kl_loss = kl.sum() / x.size(0) #divide by batch size
#
#         if self.loss == 'MSE':
#             recon = 0.5 * (x_recon - x).pow(2)
#             recon_loss = recon.sum() / x.size(0)
#         elif self.loss == 'BCE':
#             recon_loss_func = nn.BCELoss()
#             recon_loss = x.size(1)*x.size(2)*x.size(3) * recon_loss_func(x_recon,x)
#
#         loss = beta*kl_loss + recon_loss
#         #print(f'KL loss is : {kl_loss} and recon loss is : {recon_loss}')
#         return loss, kl_loss, recon_loss
