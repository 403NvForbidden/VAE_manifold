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
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
from models.nn_modules import Conv, ConvUpsampling, Skip_Conv_down, Skip_DeConv_up, Linear_block, \
    Encoder, Decoder, MLP_MI_estimator, Encoder_256, Decoder_256
import numpy as np
from models.train_net import reparameterize, kl_divergence, scalar_loss
from sklearn.mixture import GaussianMixture

from util.pytorch_msssim import ssim, msssim
from .model import AbstractModel
import pytorch_lightning as pl
import torch.optim as optim
from torch.autograd import Variable

from ._types_ import *


def weight_initialization(modules):
    ### weight initialization
    for m in modules:
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


class twoStageInfoMaxVAE(AbstractModel, pl.LightningModule):
    def __init__(self, zdim_1=100, zdim_2=3, input_channels=3, input_size=64, alpha=1, beta=1, gamma=100,
                 filepath=None):
        super().__init__(filepath=filepath)
        """
        param :
            zdim (int) : dimension of the latent space
            beta (float) : weight coefficient for DL divergence, when beta=1 is Valina VAE
            Modulate the complexity of the model with parameter 'base_enc' and 'base_dec'
        """
        self.zdim_aux = zdim_1
        self.zdim = zdim_2
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.input_channels = input_channels
        self.input_size = input_size
        self.z1 = None
        self.z2 = None
        # temp variable that saved the list of last step losses of the model
        self.history = []
        self.loss_dict = {}

        self.MI_estimator_1 = MLP_MI_estimator(input_dim=input_size * input_size * input_channels, zdim=zdim_1)
        self.MI_estimator_2 = MLP_MI_estimator(input_dim=input_size * input_size * input_channels, zdim=zdim_2)

        ##### Encoding layers #####
        self.encoder = Encoder(out_size=256, input_channel=input_channels)

        # inference mean and std
        self.mu_logvar_gen_1 = nn.Linear(256, self.zdim_aux * 2)  # 256 -> 6
        self.mu_logvar_gen_2 = nn.Linear(256, self.zdim * 2)

        # to constrain logvar in a reasonable range
        self.stabilize_exp = nn.Hardtanh(min_val=-6., max_val=2.)  # linear between min and max

        ##### Decoding layers #####
        self.decoder_1 = Decoder(zdim=zdim_1, input_channel=input_channels)
        self.decoder_2 = Decoder(zdim=zdim_2, input_channel=input_channels)

        # run weight initialization TODO: make this abstract class
        weight_initialization(self.modules())

    def encode(self, img):
        return self.encoder(img)

    def inferece_aux(self, x):
        mu_logvar_1 = self.mu_logvar_gen_1(x)
        mu_z_1, logvar_z_1 = mu_logvar_1.view(-1, self.zdim_aux, 2).unbind(-1)
        logvar_z_1 = self.stabilize_exp(logvar_z_1)
        z_1 = reparameterize(mu_z_1, logvar_z_1, self.training)
        return z_1, mu_z_1, logvar_z_1

    def inference(self, x):
        # for lower VAE
        mu_logvar_2 = self.mu_logvar_gen_2(x)
        mu_z_2, logvar_z_2 = mu_logvar_2.view(-1, self.zdim, 2).unbind(-1)
        logvar_z_2 = self.stabilize_exp(logvar_z_2)
        z_2 = reparameterize(mu_z_2, logvar_z_2, self.training)
        return z_2, mu_z_2, logvar_z_2

    def decode_aux(self, z_aux):
        return self.decoder_1(z_aux)

    def decode(self, z):
        return self.decoder_2(z)

    def forward(self, img, **kwargs):
        x_1 = self.encode(img)
        x_2 = x_1.clone()  # copy for VAE2

        z_2, mu_z_2, logvar_z_2 = self.inference(x_2)
        z_1, mu_z_1, logvar_z_1 = self.inferece_aux(x_1)

        x_recon_lo = self.decode(z_2)
        x_recon_hi = self.decode_aux(z_1)

        return x_recon_hi, mu_z_1, logvar_z_1, z_1.squeeze(), x_recon_lo, mu_z_2, logvar_z_2, z_2.squeeze()

    def step(self, img) -> dict:
        x_recon_hi, mu_z_1, logvar_z_1, z_1, x_recon_lo, mu_z_2, logvar_z_2, z_2 = self.forward(img)
        self.z1 = z_1
        self.z2 = z_2
        return self.objective_func(img, x_recon_hi, mu_z_1, logvar_z_1, z_1, x_recon_lo, mu_z_2, logvar_z_2, z_2)

    def objective_func(self, x, x_recon_hi, mu_z_1, logvar_z_1, z_1, x_recon_lo, mu_z_2, logvar_z_2, z_2, *args,
                       **kwargs) -> dict:
        """
        :param x:
        :param x_recon:
        :param mu_z:
        :param logvar_z:
        :return: the loss of beta VAEs
        """
        # loss for VAE1
        loss_recon_1 = F.mse_loss(torch.sigmoid(x_recon_hi), x)
        loss_1, loss_kl_1 = scalar_loss(x, loss_recon_1, mu_z_1, logvar_z_1, self.beta)
        MI_1 = self.MI_estimator_1(x, z_1)

        # loss for VAE2
        loss_recon_2 = F.mse_loss(torch.sigmoid(x_recon_lo), x)
        loss_2, loss_kl_2 = scalar_loss(x, loss_recon_2, mu_z_2, logvar_z_2, self.beta)
        MI_2 = self.MI_estimator_2(x, z_2)

        ## adding MI regularizer
        loss = (loss_1 - self.alpha * MI_1) + self.gamma * (loss_2 - self.alpha * MI_2)
        return {'loss': loss, 'MI_loss_1': -MI_1, 'MI_loss_2': -MI_2, 'recon_loss_1': loss_recon_1, 'KLD_1': loss_kl_1,
                'recon_loss_2': loss_recon_2, "KLD_2": loss_kl_2}

    def training_step(self, batch, batch_idx, optimizer_idx=0, *args, **kwargs):
        img, labels = batch
        self.curr_device = img.device
        # VAE 1 only step once
        if optimizer_idx == 0:
            self.loss_dict = self.step(img)
            return self.loss_dict['loss']
        elif optimizer_idx == 1:
            self.loss_dict['MI_loss_1'] = -self.MI_estimator_1(img, self.z1)
            return self.loss_dict['MI_loss_1']
        elif optimizer_idx == 2:
            # detach and add to to list
            temp_dict = {}
            for k, v in self.loss_dict.items():
                temp_dict[k] = v.clone().data.cpu()
            self.history.append(temp_dict)

            self.loss_dict['MI_loss_2'] = -self.MI_estimator_2(img, self.z2)
            return self.loss_dict['MI_loss_2']

    def backward(self, loss: Tensor, optimizer: Optimizer, optimizer_idx: int, *args, **kwargs) -> None:
        loss.backward(retain_graph=True)

    def configure_optimizers(self, params):

        optimizer_VAE_1 = optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.mu_logvar_gen_1.parameters()},
            {'params': self.mu_logvar_gen_2.parameters()},
            {'params': self.decoder_1.parameters()},
            {'params': self.decoder_2.parameters()},
        ],
            lr=params['lr'], betas=(0.9, 0.999),
            weight_decay=params['weight_decay'])

        optimizer_MI = optim.Adam([{'params': self.MI_estimator_1.parameters()}],
                                  lr=params['lr'], betas=(0.9, 0.999),
                                  weight_decay=params['weight_decay'])

        optimizer_MI_2 = optim.Adam(self.MI_estimator_2.parameters(),
                                    lr=params['lr'], betas=(0.9, 0.999),
                                    weight_decay=params['weight_decay'])

        scheduler_VAE_1 = optim.lr_scheduler.ExponentialLR(optimizer_VAE_1, gamma=params['scheduler_gamma'])
        # scheduler_VAE_2 = optim.lr_scheduler.ExponentialLR(optimizer_VAE_2, gamma=params['scheduler_gamma'])
        scheduler_MI_1 = optim.lr_scheduler.ExponentialLR(optimizer_MI, gamma=params['scheduler_gamma'])
        scheduler_MI_2 = optim.lr_scheduler.ExponentialLR(optimizer_MI_2, gamma=params['scheduler_gamma'])

        return (
            {'optimizer': optimizer_VAE_1, 'lr_scheduler': scheduler_VAE_1},
            {'optimizer': optimizer_MI, 'lr_scheduler': scheduler_MI_1},
            {'optimizer': optimizer_MI_2, 'lr_scheduler': scheduler_MI_2},
            # {'optimizer': optimizer_VAE_2, 'lr_scheduler': scheduler_VAE_1},
        )


class twoStageBetaVAE(AbstractModel, pl.LightningModule):
    def __init__(self, zdim_1=100, zdim_2=3, input_channels=3, input_size=64, alpha=1, beta=1, gamma=100,
                 filepath=None):
        super().__init__(filepath=filepath)
        """
        param :
            zdim (int) : dimension of the latent space
            beta (float) : weight coefficient for DL divergence, when beta=1 is Valina VAE
            Modulate the complexity of the model with parameter 'base_enc' and 'base_dec'
        """
        self.zdim_aux = zdim_1
        self.zdim = zdim_2
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.input_channels = input_channels
        self.input_size = input_size
        self.z1 = None
        self.z2 = None
        # temp variable that saved the list of last step losses of the model
        self.history = []
        self.loss_dict = {}

        ##### Encoding layers #####
        self.encoder = Encoder(out_size=256, input_channel=input_channels)

        # inference mean and std
        self.mu_logvar_gen_1 = nn.Linear(256, self.zdim_aux * 2)  # 256 -> 6
        self.mu_logvar_gen_2 = nn.Linear(256, self.zdim * 2)

        # to constrain logvar in a reasonable range
        self.stabilize_exp = nn.Hardtanh(min_val=-6., max_val=2.)  # linear between min and max

        ##### Decoding layers #####
        self.decoder_1 = Decoder(zdim=zdim_1, input_channel=input_channels)
        self.decoder_2 = Decoder(zdim=zdim_2, input_channel=input_channels)

        # run weight initialization TODO: make this abstract class
        weight_initialization(self.modules())

    def encode(self, img):
        return self.encoder(img)

    def inferece_aux(self, x):
        mu_logvar_1 = self.mu_logvar_gen_1(x)
        mu_z_1, logvar_z_1 = mu_logvar_1.view(-1, self.zdim_aux, 2).unbind(-1)
        logvar_z_1 = self.stabilize_exp(logvar_z_1)
        z_1 = reparameterize(mu_z_1, logvar_z_1, self.training)
        return z_1, mu_z_1, logvar_z_1

    def inference(self, x):
        # for lower VAE
        mu_logvar_2 = self.mu_logvar_gen_2(x)
        mu_z_2, logvar_z_2 = mu_logvar_2.view(-1, self.zdim, 2).unbind(-1)
        logvar_z_2 = self.stabilize_exp(logvar_z_2)
        z_2 = reparameterize(mu_z_2, logvar_z_2, self.training)
        return z_2, mu_z_2, logvar_z_2

    def decode_aux(self, z_aux):
        return self.decoder_1(z_aux)

    def decode(self, z):
        return self.decoder_2(z)

    def forward(self, img, **kwargs):
        x_1 = self.encode(img)
        x_2 = x_1.clone()  # copy for VAE2

        z_2, mu_z_2, logvar_z_2 = self.inference(x_2)
        z_1, mu_z_1, logvar_z_1 = self.inferece_aux(x_1)

        x_recon_lo = self.decode(z_2)
        x_recon_hi = self.decode_aux(z_1)

        return x_recon_hi, mu_z_1, logvar_z_1, z_1.squeeze(), x_recon_lo, mu_z_2, logvar_z_2, z_2.squeeze()

    def step(self, img) -> dict:
        x_recon_hi, mu_z_1, logvar_z_1, z_1, x_recon_lo, mu_z_2, logvar_z_2, z_2 = self.forward(img)
        self.z1 = z_1
        self.z2 = z_2
        return self.objective_func(img, x_recon_hi, mu_z_1, logvar_z_1, z_1, x_recon_lo, mu_z_2, logvar_z_2, z_2)

    def objective_func(self, x, x_recon_hi, mu_z_1, logvar_z_1, z_1, x_recon_lo, mu_z_2, logvar_z_2, z_2, *args,
                       **kwargs) -> dict:
        """
        :param x:
        :param x_recon:
        :param mu_z:
        :param logvar_z:
        :return: the loss of beta VAEs
        """
        # loss for VAE1
        loss_recon_1 = F.mse_loss(torch.sigmoid(x_recon_hi), x)
        loss_1, loss_kl_1 = scalar_loss(x, loss_recon_1, mu_z_1, logvar_z_1, self.beta)

        # loss for VAE2
        loss_recon_2 = F.mse_loss(torch.sigmoid(x_recon_lo), x)
        loss_2, loss_kl_2 = scalar_loss(x, loss_recon_2, mu_z_2, logvar_z_2, self.beta)

        ## adding MI regularizer
        loss = loss_1 + self.gamma * loss_2
        return {'loss': loss, 'recon_loss_1': loss_recon_1, 'KLD_1': loss_kl_1,
                'recon_loss_2': loss_recon_2, "KLD_2": loss_kl_2}

    def training_step(self, batch, batch_idx, optimizer_idx=0, *args, **kwargs):
        img, labels = batch
        self.curr_device = img.device
        # VAE 1 only step once
        self.loss_dict = self.step(img)
        return self.loss_dict['loss']

    def backward(self, loss: Tensor, optimizer: Optimizer, optimizer_idx: int, *args, **kwargs) -> None:
        loss.backward()

    def configure_optimizers(self, params):
        optimizer_VAE_1 = optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.mu_logvar_gen_1.parameters()},
            {'params': self.mu_logvar_gen_2.parameters()},
            {'params': self.decoder_1.parameters()},
            {'params': self.decoder_2.parameters()},
        ],
            lr=params['lr'], betas=(0.9, 0.999),
            weight_decay=params['weight_decay'])

        scheduler_VAE_1 = optim.lr_scheduler.ExponentialLR(optimizer_VAE_1, gamma=params['scheduler_gamma'])

        return (
            {'optimizer': optimizer_VAE_1, 'lr_scheduler': scheduler_VAE_1},
        )


class twoStageVaDE(AbstractModel, pl.LightningModule):
    def __init__(self, zdim_1=100, zdim_2=3, input_channels=3, input_size=64, alpha=1, beta=1, gamma=100, ydim=3,
                 filepath=None):
        super().__init__(filepath=filepath)
        """
        param :
            zdim (int) : dimension of the latent space
            beta (float) : weight coefficient for DL divergence, when beta=1 is Valina VAE
            Modulate the complexity of the model with parameter 'base_enc' and 'base_dec'
        """
        self.zdim_aux = zdim_1
        self.zdim = zdim_2
        self.ydim = ydim
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.input_channels = input_channels
        self.input_size = input_size
        self.z1 = None
        self.z2 = None
        # temp variable that saved the list of last step losses of the model
        self.history = []
        self.loss_dict = {}

        self.MI_estimator_1 = MLP_MI_estimator(input_dim=input_size * input_size * input_channels, zdim=zdim_1)
        self.MI_estimator_2 = MLP_MI_estimator(input_dim=input_size * input_size * input_channels, zdim=zdim_2)

        ### initialise GMM parameters
        # theta p
        self.pi_ = nn.Parameter(torch.FloatTensor(ydim, ).fill_(1) / ydim, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(self.zdim, ydim).fill_(0), requires_grad=True)
        # lambda p
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(self.zdim, ydim).fill_(0), requires_grad=True)

        ##### Encoding layers #####
        self.encoder = Encoder(out_size=256, input_channel=input_channels)

        # inference mean and std
        self.mu_logvar_gen_1 = nn.Linear(256, self.zdim_aux * 2)  # 256 -> 6
        self.mu_logvar_gen_2 = nn.Linear(256, self.zdim * 2)

        # to constrain logvar in a reasonable range
        self.stabilize_exp = nn.Hardtanh(min_val=-6., max_val=2.)  # linear between min and max

        ##### Decoding layers #####
        self.decoder_1 = Decoder(zdim=zdim_1, input_channel=input_channels)
        self.decoder_2 = Decoder(zdim=zdim_2, input_channel=input_channels)

        # run weight initialization TODO: make this abstract class
        weight_initialization(self.modules())

    def encode(self, img):
        return self.encoder(img)

    def inferece_aux(self, x):
        mu_logvar_1 = self.mu_logvar_gen_1(x)
        mu_z_1, logvar_z_1 = mu_logvar_1.view(-1, self.zdim_aux, 2).unbind(-1)
        logvar_z_1 = self.stabilize_exp(logvar_z_1)
        z_1 = reparameterize(mu_z_1, logvar_z_1, self.training)
        return z_1, mu_z_1, logvar_z_1

    def inference(self, x):
        # for lower VAE
        mu_logvar_2 = self.mu_logvar_gen_2(x)
        mu_z_2, logvar_z_2 = mu_logvar_2.view(-1, self.zdim, 2).unbind(-1)
        logvar_z_2 = self.stabilize_exp(logvar_z_2)
        z_2 = reparameterize(mu_z_2, logvar_z_2, self.training)
        return z_2, mu_z_2, logvar_z_2

    def decode_aux(self, z_aux):
        return self.decoder_1(z_aux)

    def decode(self, z):
        return self.decoder_2(z)

    def forward(self, img, **kwargs):
        x_1 = self.encode(img)
        x_2 = x_1.clone()  # copy for VAE2

        z_2, mu_z_2, logvar_z_2 = self.inference(x_2)
        z_1, mu_z_1, logvar_z_1 = self.inferece_aux(x_1)

        x_recon_lo = self.decode(z_2)
        x_recon_hi = self.decode_aux(z_1)

        return x_recon_hi, mu_z_1, logvar_z_1, z_1.squeeze(), x_recon_lo, mu_z_2, logvar_z_2, z_2.squeeze()

    def step(self, img) -> dict:
        x_recon_hi, mu_z_1, logvar_z_1, z_1, x_recon_lo, mu_z_2, logvar_z_2, z_2 = self.forward(img)
        self.z1 = z_1
        self.z2 = z_2
        return self.objective_func(img, x_recon_hi, mu_z_1, logvar_z_1, z_1, x_recon_lo, mu_z_2, logvar_z_2, z_2)

    def objective_func(self, x, x_recon_hi, mu_z_1, logvar_z_1, z_1, x_recon_lo, mu_z_2, logvar_z_2, z_2, *args,
                       **kwargs) -> dict:
        """
        :param x:
        :param x_recon:
        :param mu_z:
        :param logvar_z:
        :return: the loss of beta VAEs
        """
        # loss for VAE1
        loss_recon_1 = F.mse_loss(torch.sigmoid(x_recon_hi), x)
        loss_1, loss_kl_1 = scalar_loss(x, loss_recon_1, mu_z_1, logvar_z_1, self.beta)
        MI_1 = self.MI_estimator_1(x, z_1)

        # loss for VAE2
        MI_2 = self.MI_estimator_2(x, z_2)

        Z = z_2.unsqueeze(2).expand(z_2.size()[0], z_2.size()[1], self.ydim)  # NxDxK
        z_mean_t = mu_z_2.unsqueeze(2).expand(mu_z_2.size()[0], mu_z_2.size()[1], self.ydim)
        z_log_var_t = logvar_z_2.unsqueeze(2).expand(logvar_z_2.size()[0], logvar_z_2.size()[1], self.ydim)
        u_tensor3 = self.mu_c.unsqueeze(0).expand(z_2.size()[0], self.mu_c.size()[0], self.mu_c.size()[1])  # NxDxK
        lambda_tensor3 = self.log_sigma2_c.unsqueeze(0).expand(z_2.size()[0], self.log_sigma2_c.size()[0],
                                                               self.log_sigma2_c.size()[1])
        theta_tensor2 = self.pi_.unsqueeze(0).expand(z_2.size()[0], self.ydim)  # NxK

        p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5 * torch.log(2 * math.pi * lambda_tensor3) + \
                                                               (Z - u_tensor3) ** 2 / (2 * lambda_tensor3),
                                                               dim=1)) + 1e-9  # NxK
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)  # NxK

        # adding min for loss to prevent log(0)
        loss_recon_2 = F.mse_loss(torch.sigmoid(x_recon_lo), x, reduction='sum').div(
            x.size(0))  # F.binary_cross_entropy_with_logits(x_recon_lo, x, reduction='sum').div(x.size(0))

        logpzc = torch.sum(0.5 * gamma * torch.sum(self.zdim * math.log(2 * math.pi) + torch.log(lambda_tensor3) + \
                                                   torch.exp(z_log_var_t) / lambda_tensor3 + \
                                                   (z_mean_t - u_tensor3) ** 2 / lambda_tensor3, dim=1),
                           dim=-1)
        qentropy = -0.5 * torch.sum(1 + logvar_z_2 + self.zdim * math.log(2 * math.pi), -1)
        logpc = -torch.sum(torch.log(theta_tensor2) * gamma, -1)
        logqcx = torch.sum(torch.log(gamma) * gamma, -1)

        loss_kl_2 = torch.mean(logpzc + qentropy + logpc + logqcx)
        loss_2 = loss_recon_2 + self.beta * loss_kl_2
        # Normalise by same number of elements as in reconstruction

        ## adding MI regularizer
        loss = (loss_1 - self.alpha * MI_1) + self.gamma * (loss_2 - self.alpha * MI_2)
        return {'loss': loss, 'MI_loss_1': -MI_1, 'MI_loss_2': -MI_2, 'recon_loss_1': loss_recon_1, 'KLD_1': loss_kl_1,
                'recon_loss_2': loss_recon_2, "KLD_2": loss_kl_2}

    def training_step(self, batch, batch_idx, optimizer_idx=0, *args, **kwargs):
        img, labels = batch
        self.curr_device = img.device
        # VAE 1 only step once
        if optimizer_idx == 0:
            self.loss_dict = self.step(img)
            return self.loss_dict['loss']
        elif optimizer_idx == 1:
            self.loss_dict['MI_loss_1'] = -self.MI_estimator_1(img, self.z1)
            return self.loss_dict['MI_loss_1']
        elif optimizer_idx == 2:
            # detach and add to to list
            temp_dict = {}
            for k, v in self.loss_dict.items():
                temp_dict[k] = v.clone().data.cpu()
            self.history.append(temp_dict)

            self.loss_dict['MI_loss_2'] = -self.MI_estimator_2(img, self.z2)
            return self.loss_dict['MI_loss_2']

    def backward(self, loss: Tensor, optimizer: Optimizer, optimizer_idx: int, *args, **kwargs) -> None:
        loss.backward(retain_graph=True)

    def configure_optimizers(self, params):

        optimizer_VAE_1 = optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.mu_logvar_gen_1.parameters()},
            {'params': self.mu_logvar_gen_2.parameters()},
            {'params': self.decoder_1.parameters()},
            {'params': self.decoder_2.parameters()},
        ],
            lr=params['lr'], betas=(0.9, 0.999),
            weight_decay=params['weight_decay'])

        optimizer_MI = optim.Adam([{'params': self.MI_estimator_1.parameters()}],
                                  lr=params['lr'], betas=(0.9, 0.999),
                                  weight_decay=params['weight_decay'])

        optimizer_MI_2 = optim.Adam(self.MI_estimator_2.parameters(),
                                    lr=params['lr'], betas=(0.9, 0.999),
                                    weight_decay=params['weight_decay'])

        scheduler_VAE_1 = optim.lr_scheduler.ExponentialLR(optimizer_VAE_1, gamma=params['scheduler_gamma'])
        # scheduler_VAE_2 = optim.lr_scheduler.ExponentialLR(optimizer_VAE_2, gamma=params['scheduler_gamma'])
        scheduler_MI_1 = optim.lr_scheduler.ExponentialLR(optimizer_MI, gamma=params['scheduler_gamma'])
        scheduler_MI_2 = optim.lr_scheduler.ExponentialLR(optimizer_MI_2, gamma=params['scheduler_gamma'])

        return (
            {'optimizer': optimizer_VAE_1, 'lr_scheduler': scheduler_VAE_1},
            {'optimizer': optimizer_MI, 'lr_scheduler': scheduler_MI_1},
            {'optimizer': optimizer_MI_2, 'lr_scheduler': scheduler_MI_2},
            # {'optimizer': optimizer_VAE_2, 'lr_scheduler': scheduler_VAE_1},
        )


class EnhancedVAE(AbstractModel, pl.LightningModule):
    def __init__(self, zdim=3, input_channels=3, input_size=64, beta=1, filepath=None):
        super().__init__(filepath=filepath)
        """
        param :
            params: 
            zdim (int) : dimension of the latent space
            beta (float) : weight coefficient for DL divergence, when beta=1 is Valina VAE
            Modulate the complexity of the model with parameter 'base_enc' and 'base_dec'
        """
        self.zdim = zdim
        self.beta = beta
        self.input_channels = input_channels
        self.input_size = input_size
        self.history = []
        ##### Encoding layers #####
        self.encoder = Encoder_256(out_size=256, input_channel=input_channels)

        # inference mean and std
        self.mu_logvar_gen = nn.Linear(256, self.zdim * 2)  # 256 -> 6

        # to constrain logvar in a reasonable range
        self.stabilize_exp = nn.Hardtanh(min_val=-6., max_val=2.)  # linear between min and max

        ##### Decoding layers #####
        self.decoder = Decoder_256(zdim=zdim, input_channel=input_channels)

        # run weight initialization TODO: make this abstract class
        weight_initialization(self.modules())

    ### @override
    def encode(self, img):
        return self.encoder(img)

    ### @override
    def inference(self, mu_logvar):
        mu_logvar = self.mu_logvar_gen(mu_logvar)
        mu_z, logvar_z = mu_logvar.view(-1, self.zdim, 2).unbind(-1)
        logvar_z = self.stabilize_exp(logvar_z)
        return reparameterize(mu_z, logvar_z, self.training), mu_z, logvar_z

    ### @override
    def decode(self, z):
        return self.decoder(z)

    ### @override
    def training_step(self, batch, batch_idx, optimizer_idx=0, *args, **kwargs) -> dict:
        img, labels = batch
        self.curr_device = img.device
        # only forward once
        return self.step(img)

    ### @override
    def forward(self, img, **kwargs):
        # (32, 3, 64, 64) or (3, 100)
        x = self.encode(img)
        z, mu_z, logvar_z = self.inference(x)
        x_recon = self.decode(z)
        return x_recon, mu_z, logvar_z, z.squeeze()

    def step(self, img) -> dict:
        x_recon, mu_z, logvar_z, _ = self.forward(img)
        loss = self.objective_func(img, x_recon, mu_z, logvar_z)
        return loss

    ### @override
    def backward(self, loss: Tensor, optimizer: Optimizer, optimizer_idx: int, *args, **kwargs) -> None:
        loss.backward()

    ### @override
    def objective_func(self, x, x_recon, mu_z, logvar_z, *args,
                       **kwargs) -> dict:
        """
        :param x:
        :param x_recon:
        :param mu_z:
        :param logvar_z:
        :return: the loss of beta VAEs
        """
        # x_recon = torch.sigmoid(x_recon)

        # loss_recon = (1 - msssim(x, x_recon, normalize='relu')) + F.l1_loss(x_recon, x, reduction='sum').div(x.size(0))
        # print(msssim(x, x_recon, size_average=False))
        # loss_kl = kl_divergence(mu_z, logvar_z)
        # loss = loss_recon + self.beta * loss_kl

        # loss_recon = F.mse_loss( x_recon, x, reduction='sum').div(x.size(0))
        loss_recon = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(x.size(0))
        loss_kl = kl_divergence(mu_z, logvar_z)
        loss = loss_recon + self.beta * loss_kl

        # print(loss.item(), loss_kl.item(), loss_recon.item())
        return {'loss': loss, 'recon_loss': loss_recon, 'KLD': loss_kl}

    ### @override
    def configure_optimizers(self, params: dict) -> dict:
        optimizer = optim.Adam(self.parameters(),
                               lr=params['lr'], betas=(0.9, 0.999),
                               weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=params['scheduler_gamma'])

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class betaVAE(AbstractModel, pl.LightningModule):
    def __init__(self, zdim=3, input_channels=3, input_size=64, beta=1, filepath=None):
        super().__init__(filepath=filepath)
        """
        param :
            params: 
            zdim (int) : dimension of the latent space
            beta (float) : weight coefficient for DL divergence, when beta=1 is Valina VAE
            Modulate the complexity of the model with parameter 'base_enc' and 'base_dec'
        """
        self.zdim = zdim
        self.beta = beta
        self.input_channels = input_channels
        self.input_size = input_size
        self.history = []
        ##### Encoding layers #####
        self.encoder = Encoder(out_size=256, input_channel=input_channels)

        # inference mean and std
        self.mu_logvar_gen = nn.Linear(256, self.zdim * 2)  # 256 -> 6

        # to constrain logvar in a reasonable range
        self.stabilize_exp = nn.Hardtanh(min_val=-6., max_val=2.)  # linear between min and max

        ##### Decoding layers #####
        self.decoder = Decoder(zdim=zdim, input_channel=input_channels)

        # run weight initialization TODO: make this abstract class
        weight_initialization(self.modules())

    ### @override
    def encode(self, img):
        return self.encoder(img)

    ### @override
    def inference(self, mu_logvar):
        mu_logvar = self.mu_logvar_gen(mu_logvar)
        mu_z, logvar_z = mu_logvar.view(-1, self.zdim, 2).unbind(-1)
        logvar_z = self.stabilize_exp(logvar_z)
        return reparameterize(mu_z, logvar_z, self.training), mu_z, logvar_z

    ### @override
    def decode(self, z):
        return self.decoder(z)

    ### @override
    def training_step(self, batch, batch_idx, optimizer_idx=0, *args, **kwargs) -> dict:
        img, labels = batch
        self.curr_device = img.device
        # only forward once
        return self.step(img)

    ### @override
    def forward(self, img, **kwargs):
        # (32, 3, 64, 64) or (3, 100)
        x = self.encode(img)
        z, mu_z, logvar_z = self.inference(x)
        x_recon = self.decode(z)
        return x_recon, mu_z, logvar_z, z.squeeze()

    def step(self, img) -> dict:
        x_recon, mu_z, logvar_z, _ = self.forward(img)
        loss = self.objective_func(img, x_recon, mu_z, logvar_z)
        return loss

    ### @override
    def backward(self, loss: Tensor, optimizer: Optimizer, optimizer_idx: int, *args, **kwargs) -> None:
        loss.backward()

    ### @override
    def objective_func(self, x, x_recon, mu_z, logvar_z, *args,
                       **kwargs) -> dict:
        """
        :param x:
        :param x_recon:
        :param mu_z:
        :param logvar_z:
        :return: the loss of beta VAEs
        """

        # x_recons = torch.sigmoid(x_recons)
        # recons_loss = F.mse_loss(
        #     x_recons, x, reduction='sum'
        # ).div(batch_size)

        loss_recon = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(x.size(0))
        # loss_recon = F.mse_loss(torch.sigmoid(x_recon), x)
        # loss, loss_kl = scalar_loss(x, loss_recon, mu_z, logvar_z, self.beta)

        loss_kl = kl_divergence(mu_z, logvar_z)
        loss = loss_recon + self.beta * loss_kl

        # if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
        #     loss = recons_loss + self.beta * kld_weight * kld_loss
        # elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
        #     self.C_max = self.C_max.to(input.device)
        #     C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
        #     loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        # else:
        #     raise ValueError('Undefined loss type.')

        return {'loss': loss, 'recon_loss': loss_recon, 'KLD': loss_kl}
        # TODO: sample function
        # TODO: tensorboard add image follow this tutorial: https://learnopencv.com/tensorboard-with-pytorch-lightning/

    ### @override
    def configure_optimizers(self, params: dict) -> dict:
        optimizer = optim.Adam(self.parameters(),
                               lr=params['lr'], betas=(0.9, 0.999),
                               weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=params['scheduler_gamma'])

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class infoMaxVAE(AbstractModel, pl.LightningModule):
    def __init__(self, zdim=3, input_channels=3, input_size=64, alpha=1, beta=1, filepath=None):
        super().__init__(filepath=filepath)
        """
        param :
            zdim (int) : dimension of the latent space
            beta (float) : weight coefficient for DL divergence, when beta=1 is Valina VAE
            Modulate the complexity of the model with parameter 'base_enc' and 'base_dec'
        """
        self.zdim = zdim
        self.beta = beta
        self.alpha = alpha
        self.input_channels = input_channels
        self.input_size = input_size
        self.MI_estimator = MLP_MI_estimator(input_dim=input_size * input_size * input_channels, zdim=zdim)
        self.VAE = betaVAE(zdim, input_channels, input_size, beta=beta)
        self.z = None
        # temp variable that saved the list of last step losses of the model
        self.history = []
        self.loss_dict = {}
        weight_initialization(self.modules())

    def encode(self, x):
        return self.VAE.encoder(x)

    def inference(self, x):
        return self.VAE.inference(x)

    def decode(self, z):
        return self.VAE.decoder(z)

    def forward(self, img, **kwargs):
        return self.VAE(img)

    def step(self, img) -> dict:
        x_recon, mu_z, logvar_z, z = self.forward(img)
        self.z = z
        loss = self.objective_func(img, x_recon, mu_z, logvar_z, z)
        return loss

    def objective_func(self, x, x_recon, mu_z, logvar_z, z, *args,
                       **kwargs) -> dict:
        """
        :param x:
        :param x_recon:
        :param mu_z:
        :param logvar_z:
        :return: the loss of beta VAEs
        """
        loss_recon = F.binary_cross_entropy_with_logits(x_recon, x)
        loss, loss_kl = scalar_loss(x, loss_recon, mu_z, logvar_z, self.beta)

        MI_loss = -self.MI_estimator(x, z)
        ## adding MI regularizer
        loss += self.alpha * MI_loss

        return {'loss': loss, 'MI_loss': MI_loss, 'recon_loss': loss_recon, 'KLD': loss_kl}

    def training_step(self, batch, batch_idx, optimizer_idx=0, *args, **kwargs):
        img, labels = batch
        self.curr_device = img.device
        if optimizer_idx == 0:
            self.loss_dict = self.step(img)
            return self.loss_dict['loss']
        elif optimizer_idx == 1:
            self.loss_dict['MI_loss'] = -self.MI_estimator(img, self.z)

            # detach and add to to list
            temp_dict = {}
            for k, v in self.loss_dict.items():
                temp_dict[k] = v.clone().data.cpu()
            self.history.append(temp_dict)
            return self.loss_dict['MI_loss']

    def backward(self, loss: Tensor, optimizer: Optimizer, optimizer_idx: int, *args, **kwargs) -> None:
        if optimizer_idx == 0:
            loss.backward(retain_graph=True)
        else:
            loss.backward()

    def configure_optimizers(self, params):
        optimizer_VAE = optim.Adam([{'params': self.VAE.parameters()}],
                                   lr=params['lr'], betas=(0.9, 0.999),
                                   weight_decay=params['weight_decay'])
        optimizer_MI = optim.Adam(self.MI_estimator.parameters(),
                                  lr=params['lr'] * 100, betas=(0.9, 0.999),
                                  weight_decay=params['weight_decay'])

        scheduler_VAE = optim.lr_scheduler.ExponentialLR(optimizer_VAE, gamma=params['scheduler_gamma'])
        scheduler_MI = optim.lr_scheduler.ExponentialLR(optimizer_MI, gamma=params['scheduler_gamma'])

        return (
            {'optimizer': optimizer_VAE, 'lr_scheduler': scheduler_VAE},
            {'optimizer': optimizer_MI, 'lr_scheduler': scheduler_MI},
        )


class VaDE(AbstractModel, pl.LightningModule):
    def __init__(self, zdim=3, ydim=7, input_channels=3, input_size=64, filepath=None):
        super().__init__(filepath=filepath)
        """
        https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch/blob/master/udlp/clustering/vade.py
        param :
            zdim (int) : dimension of the latent space
            Modulate the complexity of the model with parameter 'base_enc' and 'base_dec'
        """
        self.zdim = zdim
        self.ydim = ydim
        self.input_channels = input_channels
        self.input_size = input_size
        self.history = []

        ### initialise GMM parameters
        # theta p
        self.pi_ = nn.Parameter(torch.FloatTensor(ydim, ).fill_(1) / ydim, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(zdim, ydim).fill_(0), requires_grad=True)
        # lambda p
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(zdim, ydim).fill_(0), requires_grad=True)

        ###########################
        ##### Encoding layers #####
        ###########################

        self.encoder = Encoder_256(out_size=256, input_channel=input_channels) \
            if input_size == 256 else \
            Encoder(out_size=256, input_channel=input_channels)

        self.mu_l = nn.Linear(256, zdim)
        self.logvar_l = nn.Linear(256, zdim)

        # to constrain logvar in a reasonable range
        self.stabilize_exp = nn.Hardtanh(min_val=-6., max_val=2.)  # linear between min and max

        ###########################
        ##### Decoding layers #####
        ###########################
        self.decoder = Decoder_256(zdim=zdim, input_channel=input_channels) \
            if input_size == 256 else \
            Decoder(zdim=zdim, input_channel=input_channels)

        ### weight initialization
        weight_initialization(self.modules())

    @property
    def cluster_weights(self):
        return torch.softmax(self.pi_, dim=0)

    ### @override
    def encode(self, img):
        return self.encoder(img)

    ### @override
    def inference(self, mu_logvar):
        mu_z, logvar_z = self.mu_l(mu_logvar), self.logvar_l(mu_logvar)

        logvar_z = self.stabilize_exp(logvar_z)
        return reparameterize(mu_z, logvar_z, self.training), mu_z, logvar_z

    ### @override
    def decode(self, z):
        return self.decoder(z)

    ### @override
    def training_step(self, batch, batch_idx, optimizer_idx=0, *args, **kwargs) -> dict:
        img, labels = batch
        self.curr_device = img.device
        # only forward once
        return self.step(img)

    ### @override
    def forward(self, x):
        ### encode
        z, mu_z, logvar_z = self.inference(self.encode(x))
        ### decode
        x_recon = self.decode(z)
        return x_recon, mu_z, logvar_z, z.squeeze()

    def step(self, img) -> dict:
        x_recon, mu_z, logvar_z, z = self.forward(img)
        loss = self.objective_func(img, x_recon, z, mu_z, logvar_z)
        return loss

    ### @override
    def backward(self, loss: Tensor, optimizer: Optimizer, optimizer_idx: int, *args, **kwargs) -> None:
        loss.backward()

    def objective_func(self, x, recon_x, z, z_mean, z_log_var):
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
        loss_recon = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum').div(x.size(0))

        logpzc = torch.sum(0.5 * gamma * torch.sum(self.zdim * math.log(2 * math.pi) + torch.log(lambda_tensor3) + \
                                                   torch.exp(z_log_var_t) / lambda_tensor3 + \
                                                   (z_mean_t - u_tensor3) ** 2 / lambda_tensor3, dim=1),
                           dim=-1)
        qentropy = -0.5 * torch.sum(1 + z_log_var + self.zdim * math.log(2 * math.pi), -1)
        logpc = -torch.sum(torch.log(theta_tensor2) * gamma, -1)
        logqcx = torch.sum(torch.log(gamma) * gamma, -1)

        KL_loss = torch.mean(logpzc + qentropy + logpc + logqcx)
        # Normalise by same number of elements as in reconstruction
        loss = loss_recon + KL_loss

        return {'loss': loss, 'recon_loss': loss_recon, 'KLD': KL_loss}  # , 'gamma': gamma}

    # ### @override
    # def objective_func(self, x, recon_x, z, z_mean, z_log_var):
    #     Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.ydim)  # NxDxK
    #     z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.ydim)
    #     z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.ydim)
    #     u_tensor3 = self.mu_c.unsqueeze(0).expand(z.size()[0], self.mu_c.size()[0], self.mu_c.size()[1])  # NxDxK
    #     lambda_tensor3 = self.log_sigma2_c.unsqueeze(0).expand(z.size()[0], self.log_sigma2_c.size()[0],
    #                                                            self.log_sigma2_c.size()[1])
    #     theta_tensor2 = self.pi_.unsqueeze(0).expand(z.size()[0], self.ydim)  # NxK
    #
    #     p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5 * torch.log(2 * math.pi * lambda_tensor3) + \
    #                                                            (Z - u_tensor3) ** 2 / (2 * lambda_tensor3),
    #                                                            dim=1)) + 1e-9  # NxK
    #     gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)  # NxK
    #
    #     # adding min for loss to prevent log(0)
    #     BCE_withlogits = -torch.sum(x * torch.log(torch.clamp(torch.sigmoid(recon_x), min=1e-10)) +
    #                                 (1 - x) * torch.log(torch.clamp(1 - torch.sigmoid(recon_x), min=1e-10)), 1).sum(
    #         1).sum(1)
    #     # BCE_withlogits = F.binary_cross_entropy_with_logits(recon_x, x)
    #
    #     logpzc = torch.sum(0.5 * gamma * torch.sum(self.zdim * math.log(2 * math.pi) + torch.log(lambda_tensor3) + \
    #                                                torch.exp(z_log_var_t) / lambda_tensor3 + \
    #                                                (z_mean_t - u_tensor3) ** 2 / lambda_tensor3, dim=1),
    #                        dim=-1)
    #     qentropy = -0.5 * torch.sum(1 + z_log_var + self.zdim * math.log(2 * math.pi), -1)
    #     logpc = -torch.sum(torch.log(theta_tensor2) * gamma, -1)
    #     logqcx = torch.sum(torch.log(gamma) * gamma, -1)
    #
    #     KL_loss = logpzc + qentropy + logpc + logqcx
    #     # Normalise by same number of elements as in reconstruction
    #     loss = torch.mean(BCE_withlogits + KL_loss)
    #
    #     return {'loss': loss, 'recon_loss': torch.mean(BCE_withlogits), 'KLD': torch.mean(KL_loss), 'gamma': gamma}

    ### @override
    def configure_optimizers(self, params: dict) -> dict:
        optimizer = optim.Adam(self.parameters(),
                               lr=params['lr'], betas=(0.9, 0.999),
                               weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=params['scheduler_gamma'])

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
