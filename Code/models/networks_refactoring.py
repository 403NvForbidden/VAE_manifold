# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-02T12:15:22+11:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-31T10:40:55+10:00

"""
Vanilla VAE and SC-VAE

File containing different VAE architectures, to reconstruct CxHxW single cell images,
while learning a 3D latent space, for visualization purpose and downstream data interrogation.
"""

import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from models.nn_modules import Conv, ConvUpsampling, Skip_Conv_down, Skip_DeConv_up, Linear_block, \
    Encoder, Decoder
import numpy as np
from models.train_net import reparameterize, kl_divergence
from sklearn.mixture import GaussianMixture
from .model import AbstractModel


# class twoStageVAE():

class VAE2(nn.Module):
    def __init__(self, VAE1_conv_encoder, VAE1_linear_encoder, zdim=3, alpha=1, beta=1, input_channels=3,
                 base_dec_size=32, loss='BCEWithLogitsLoss', double_embed=False, conditional=False):
        super(VAE2, self).__init__()
        """
        param:
            VAE2_encoder (nn.Sequential): the encoder part of VAE_1
            zdim (int): the final target dimension default 3
            beta (float): the weight coefficient of KL loss

            Modulate the complexity of the model with parameter 'base_enc' and 'base_dec'
        """

        self.zdim = zdim
        self.input_channels = input_channels
        self.alpha = alpha
        self.beta = beta
        self.loss = loss
        self.base_dec = base_dec_size
        self.double_embed = double_embed
        self.conditional = conditional

        ###########################
        ##### Encoding layers #####
        ###########################
        # conv_enc and linear_encoder is passed from the
        if not double_embed:
            self.conv_enc = VAE1_conv_encoder
            self.linear_enc = nn.Sequential(
                VAE1_linear_encoder,  # 2*2*base_enc*16 -> 256
                nn.Linear(256, 64),  # 256 -> 64
                nn.BatchNorm1d(64),
                nn.ReLU()
            )
        else:
            # double embed
            assert input_channels == 100
            self.linear_enc = nn.Sequential(
                nn.Linear(100, 64),  # 256 -> 64
                nn.BatchNorm1d(64),
                nn.ReLU()
            )

        ###########################
        #####   Inference     #####
        ###########################
        self.mu_logvar_gen = nn.Linear(64, self.zdim * 2)  # 64 -> 6

        ###########################
        ##### Decoding layers #####
        ###########################
        # 3 -> 64 -> 256 -> 1024 -> 2048 -> 1024*2*2 -> 512*4*4 -> 256*8*8 -> 128*16*16 -> 64*32*32 -> 4*64*64a
        self.decoder = Decoder(zdim=self.zdim, input_channel=3)
        ### to constrain logvar in a reasonable range
        self.stabilize_exp = nn.Hardtanh(min_val=-6., max_val=2.)  # linear between min and max

    def initialization(self):
        ### weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # Gain adapted for LeakyReLU activation function
                xavier_normal_(m.weight, gain=math.sqrt(2. / (1 + 0.01)))
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                xavier_normal_(m.weight, gain=math.sqrt(2. / (1 + 0.01)))
                m.bias.data.fill_(0.01)

    def encode(self, x):
        # the input x here should be xxx

        batch_size = x.size(0)
        # x is image if not double embedding
        if not self.double_embed:
            x = self.conv_enc(x)
            x = x.view((batch_size, -1))
        x = self.linear_enc(x)
        mu_logvar = self.mu_logvar_gen(x)
        mu_z, logvar_z = mu_logvar.view(-1, self.zdim, 2).unbind(-1)

        logvar_z = self.stabilize_exp(logvar_z)
        z = self.reparameterize(mu_z, logvar_z), mu_z, logvar_z
        return z, mu_z, logvar_z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, input):
        # (32, 3, 64, 64) or (3, 100)
        z, mu_z, logvar_z = self.encode(input)
        x_recon = self.decode(z)
        return x_recon, mu_z, logvar_z, z.squeeze()


class betaVAE(AbstractModel):
    def __init__(self, zdim=3, input_channels=3, beta=1, filepath=None):
        super().__init__(filepath=filepath)
        """
        param :
            zdim (int) : dimension of the latent space
            beta (float) : weight coefficient for DL divergence, when beta=1 is Valina VAE
            Modulate the complexity of the model with parameter 'base_enc' and 'base_dec'
        """
        self.zdim = zdim
        self.beta = beta
        self.input_channels = input_channels

        ##### Encoding layers #####
        self.encoder = Encoder(out_size=256, input_channel=input_channels)

        # inference mean and std
        self.mu_logvar_gen = nn.Linear(256, self.zdim * 2)  # 256 -> 6

        # to constrain logvar in a reasonable range
        self.stabilize_exp = nn.Hardtanh(min_val=-6., max_val=2.)  # linear between min and max

        ##### Decoding layers #####
        self.decoder = Decoder(zdim=zdim, input_channel=input_channels)

        # run weight initialization TODO: make this abstract class
        self.initialization()

    def initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_normal_(m.weight,
                               gain=math.sqrt(2. / (1 + 0.01)))  # Gain adapted for LeakyReLU activation function
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        return self.encoder(x)

    def inference(self, x):
        mu_logvar = self.mu_logvar_gen(x)
        mu_z, logvar_z = mu_logvar.view(-1, self.zdim, 2).unbind(-1)
        logvar_z = self.stabilize_exp(logvar_z)
        return reparameterize(mu_z, logvar_z, self.training), mu_z, logvar_z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, img, **kwargs):
        # (32, 3, 64, 64) or (3, 100)
        x = self.encode(img)
        z, mu_z, logvar_z = self.inference(x)
        x_recon = self.decode(z)
        return x_recon, mu_z, logvar_z, z.squeeze()

    def objective_func(self, x, x_recon, mu_z, logvar_z, *args,
                      **kwargs) -> dict:
        """
        :param x:
        :param x_recon:
        :param mu_z:
        :param logvar_z:
        :return: the loss of beta VAEs
        """
        loss_recon = F.mse_loss(torch.sigmoid(x_recon), x)
        loss_recon *= x.size(1) * x.size(2) * x.size(3)
        loss_recon.div(x.size(0))

        loss_kl = kl_divergence(mu_z, logvar_z)

        # if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
        #     loss = recons_loss + self.beta * kld_weight * kld_loss
        # elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
        #     self.C_max = self.C_max.to(input.device)
        #     C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
        #     loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        # else:
        #     raise ValueError('Undefined loss type.')

        return {'loss': loss_recon + self.beta * loss_kl, 'recon_loss': loss_recon, 'KLD': loss_kl}
        # TODO: sample image

"""
class TwoStageVAE(AbstractModel):
    def __init__(self, z1_dim=100, z2_dim=3, input_channel=3, base_enc=32, base_dec=32, doubleEmbed=False, filepath=None):
        super(AbstractModel, self).__init__(filepath)
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.doubleEmbed=doubleEmbed

        self.VAE1 = betaVAE(zdim=z1_dim, input_channels=input_channel)
        self.linEnc = nn.Sequential(
            Linear_block(zdim, 256),
            Linear_block(256, 1024),
            Linear_block(1024, pow(2, input_channel - 1) * base_dec * 16),
        )
        self.decoder2 = Decoder(zdim=z2_dim, input_channel=input_channel)

    def forward(self, x):
        if self.doubleEmbed:

        else:
            x_encoded = self.VAE1.encode(x)
"""


class VaDE(nn.Module):
    def __init__(self, zdim=3, ydim=7, input_channels=3, base_enc=32, base_dec=32,
                 loss='BCEWithLogitsLoss'):
        super(VaDE, self).__init__()
        """
        param :
            zdim (int) : dimension of the latent space
            beta (float) : weight coefficient for DL divergence, when beta=1 is Valina VAE

            Modulate the complexity of the model with parameter 'base_enc' and 'base_dec'
        """
        self.zdim = zdim
        self.ydim = ydim
        self.loss = loss
        self.input_channels = input_channels
        self.base_dec = base_dec

        ### initialise GMM parameters
        # theta p
        self.pi_ = nn.Parameter(torch.FloatTensor(ydim, ).fill_(1) / ydim, requires_grad=True)
        # self.mu_c = nn.Parameter(torch.randn(zdim, ydim), requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(zdim, ydim).fill_(0), requires_grad=True)
        # lambda p
        # self.log_sigma2_c = nn.Parameter(torch.randn(zdim, ydim), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(zdim, ydim).fill_(0), requires_grad=True)

        ###########################
        ##### Encoding layers #####
        ###########################
        self.conv_enc = nn.Sequential(
            Conv(self.input_channels, base_enc, 4, stride=2, padding=1),  # stride 2, resolution is splitted by half
            Conv(base_enc, base_enc * 2, 4, stride=2, padding=1),  # 16x16
            Conv(base_enc * 2, base_enc * 4, 4, stride=2, padding=1),  # 8x8
            Conv(base_enc * 4, base_enc * 8, 4, stride=2, padding=1),  # 4x4
            Conv(base_enc * 8, base_enc * 16, 4, stride=2, padding=1),  # 2x2
            # Conv(base_enc * 16, base_enc * 16, 4, stride=2, padding=1),  # 2
        )
        self.linear_enc = nn.Sequential(
            Linear_block(pow(2, input_channels - 1) * base_dec * 16, 1024),  # 2048 -> 1024
            Linear_block(1024, 256),
            # nn.Linear(256, self.zdim * 2)
        )
        self.mu_l = nn.Linear(256, zdim)
        self.logvar_l = nn.Linear(256, zdim)

        # to constrain logvar in a reasonable range
        self.stabilize_exp = nn.Hardtanh(min_val=-6., max_val=2.)  # linear between min and max

        ###########################
        ##### Decoding layers #####
        ###########################
        self.decoder = Decoder(zdim, input_channel=input_channels)

        ### weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_normal_(m.weight,
                               gain=math.sqrt(2. / (1 + 0.01)))  # Gain adapted for LeakyReLU activation function
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def cluster_weights(self):
        return torch.softmax(self.pi_, dim=0)

    def encode(self, img):
        batch_size = img.size(0)
        # encode
        x = self.conv_enc(img)
        x = x.view((batch_size, -1))
        mu_logvar = self.linear_enc(x)
        mu_z, logvar_z = self.mu_l(mu_logvar), self.logvar_l(mu_logvar)

        logvar_z = self.stabilize_exp(logvar_z)
        return mu_z, logvar_z

    def forward(self, x):
        ### encode
        mu_z, logvar_z = self.encode(x)
        ### inference
        z = reparameterize(mu_z, logvar_z)
        ### decode
        x_recon = self.decoder(z)
        return x_recon, mu_z, logvar_z, z

    def loss_function(self, x, recon_x, z, z_mean, z_log_var):
        Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.ydim)  # NxDxK
        z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.ydim)
        z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.ydim)
        u_tensor3 = self.mu_c.unsqueeze(0).expand(z.size()[0], self.mu_c.size()[0], self.mu_c.size()[1])  # NxDxK
        lambda_tensor3 = self.log_sigma2_c.unsqueeze(0).expand(z.size()[0], self.log_sigma2_c.size()[0],
                                                               self.log_sigma2_c.size()[1])
        theta_tensor2 = self.pi_.unsqueeze(0).expand(z.size()[0], self.ydim)  # NxK

        p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5 * torch.log(2 * math.pi * lambda_tensor3) + \
                                                               (Z - u_tensor3) ** 2 / (2 * lambda_tensor3),
                                                               dim=1)) + 1e-9  # NxK
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)  # NxK

        # adding min for loss to prevent log(0)
        BCE_withlogits = -torch.sum(x * torch.log(torch.clamp(torch.sigmoid(recon_x), min=1e-10)) +
                                    (1 - x) * torch.log(torch.clamp(1 - torch.sigmoid(recon_x), min=1e-10)), 1).sum(
            1).sum(1)
        # BCE_withlogits = F.binary_cross_entropy_with_logits(recon_x, x)

        logpzc = torch.sum(0.5 * gamma * torch.sum(self.zdim * math.log(2 * math.pi) + torch.log(lambda_tensor3) + \
                                                   torch.exp(z_log_var_t) / lambda_tensor3 + \
                                                   (z_mean_t - u_tensor3) ** 2 / lambda_tensor3, dim=1),
                           dim=-1)
        qentropy = -0.5 * torch.sum(1 + z_log_var + self.zdim * math.log(2 * math.pi), -1)
        logpc = -torch.sum(torch.log(theta_tensor2) * gamma, -1)
        logqcx = torch.sum(torch.log(gamma) * gamma, -1)

        KL_loss = logpzc + qentropy + logpc + logqcx
        # Normalise by same number of elements as in reconstruction
        loss = torch.mean(BCE_withlogits + KL_loss)

        return loss, torch.mean(BCE_withlogits), torch.mean(KL_loss), gamma


class Skip_VAE(nn.Module):

    def __init__(self, zdim, input_channels=3, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2):
        """
        Modulate the complexity of the model with parameter 'base_enc' and 'base_dec'
        """
        super(Skip_VAE, self).__init__()
        self.zdim = zdim
        self.beta = beta
        self.input_channels = input_channels

        self.encoder = nn.Sequential(
            Skip_Conv_down(self.input_channels, base_enc),  # resolution is splitted by half
            Skip_Conv_down(base_enc, base_enc * 2, kernel_size=4, stride=2, padding=1),
            Skip_Conv_down(base_enc * 2, base_enc * 4, kernel_size=4, stride=2, padding=1),
            Skip_Conv_down(base_enc * 4, base_enc * 8, kernel_size=4, stride=2, padding=1),
            Skip_Conv_down(base_enc * 8, base_enc * 16, kernel_size=4, stride=2, padding=1),  # HxW is 2x2
            # Skip_Conv_down(base_enc*16,base_enc*16,kernel_size=4, stride=2, padding=1), ADD THIS LAYER IN INPUT IS 128x128
            Skip_Conv_down(base_enc * 16, 2 * zdim, kernel_size=2, stride=1, padding=0, LastLayer=True)
            # Equivalent to a MLP
        )

        self.decoder = nn.Sequential(
            Skip_Conv_down(zdim, int(base_dec * (depth_factor_dec ** 4)), kernel_size=1, stride=1, padding=0, factor=1,
                           mode='nearest'),
            # equivalent to a MLP
            # Skip_DeConv_up(int(base_dec*(depth_factor_dec**4)),int(base_dec*(depth_factor_dec**4))), ADD THIS LAYER IS INPUT IS 128x128
            Skip_DeConv_up(int(base_dec * (depth_factor_dec ** 4)), int(base_dec * (depth_factor_dec ** 3))),
            Skip_DeConv_up(int(base_dec * (depth_factor_dec ** 3)), int(base_dec * (depth_factor_dec ** 2))),
            Skip_DeConv_up(int(base_dec * (depth_factor_dec ** 2)), int(base_dec * (depth_factor_dec ** 1))),
            Skip_DeConv_up(int(base_dec * (depth_factor_dec ** 1)), int(base_dec * (depth_factor_dec ** 0))),
            Skip_DeConv_up(int(base_dec * (depth_factor_dec ** 0)), int(base_dec * (depth_factor_dec ** 0))),
            Skip_DeConv_up(int(base_dec * (depth_factor_dec ** 0)), self.input_channels, LastLayer=True)
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
        learnt_stats = self.encoder(x)  # Mu encode in the first half channels
        mu_z = learnt_stats[:, self.zdim:]
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


class Simple_VAE(nn.Module):
    def __init__(self, zdim=3, input_channels=1, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2, loss='BCE'):
        super(Simple_VAE, self).__init__()
        """
        Simple Shallow VAE, only used for exploring User Feedbacks possibilities on a simple dataset
        param :
            zdim (int) : dimension of the latent space

            Modulate the complexity of the model with parameter 'base_enc' and 'base_dec'
        """

        self.zdim = zdim
        self.beta = beta
        self.loss = loss
        self.input_channels = input_channels
        self.base_dec = base_dec

        ###########################
        ##### Encoding layers #####
        ###########################
        self.conv_enc = nn.Sequential(
            Conv(self.input_channels, base_enc, 4, stride=2, padding=1),  # stride 2, resolution is splitted by half
            Conv(base_enc, base_enc, 4, stride=2, padding=1),  # 16x16
            Conv(base_enc, base_enc, 4, stride=2, padding=1),  # 8x8
            Conv(base_enc, base_enc, 4, stride=2, padding=1),  # 4x4
            Conv(base_enc, base_enc, 4, stride=2, padding=1),  # 2x2
        )
        self.linear_enc = nn.Sequential(
            nn.Linear(2 * 2 * base_enc, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.mu_logvar_gen = nn.Linear(256, self.zdim * 2)

        ###########################
        ##### Decoding layers #####
        ###########################
        self.linear_dec = nn.Sequential(
            nn.Linear(self.zdim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2 * 2 * base_dec),
            nn.BatchNorm1d(2 * 2 * base_dec),
            nn.ReLU()
        )

        self.conv_dec = nn.Sequential(
            ConvUpsampling(base_dec, base_dec, 4, stride=2, padding=1),  # 4
            ConvUpsampling(base_dec, base_dec, 4, stride=2, padding=1),  # 8
            ConvUpsampling(base_dec, base_dec, 4, stride=2, padding=1),  # 16
            ConvUpsampling(base_dec, base_dec, 4, stride=2, padding=1),  # 32
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
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
        x = x.view((batch_size, self.base_dec, 2, 2))
        x_recon = self.conv_dec(x)

        return x_recon

    def forward(self, x):

        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        x_recon = self.decode(z)
        return x_recon, mu_z, logvar_z, z.squeeze()
