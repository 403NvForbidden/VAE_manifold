# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-06T12:15:33+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-31T10:26:36+10:00

"""
File containing main function to train the different VAE models with proper single cell images dataset

####### 1 #######
train(), test() and train_VAE_model() :
--> To train VAE with ELBO objective function (Vanilla VAE and SCVAE)

####### 2 #######
train_infoM_epoch(), test_infoM_epoch() and train_InfoMAX_model() :
--> To train InfoMAX VAE, jointly optimizing VAE and MLP MI estimator

####### 3 #######
train_feedback(), train_Simple_VAE() :
--> Train a VAE with human feedback stored in a CSV file
"""

import pandas as pd
import numpy as np
from timeit import default_timer as timer
from tqdm import tqdm
import os
import warnings
import itertools

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch import cuda
from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from util.helpers import plot_latent_space, show, EarlyStopping, save_brute, make_path
from util.pytorch_msssim import msssim
from .nn_modules import infoNCE_bound

from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.mixture import GaussianMixture

import pytorch_lightning as pl
from .model import AbstractModel
from ._types_ import *
import collections, functools, operator


def kl_divergence(mu, logvar):
    return torch.mean(-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(dim=1))


def reparameterize(mu, logvar, training=True):
    """Reparameterization trick.
    """
    if training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    else:  # val
        return mu


def adjust_learning_rate(init_lr, optimizer, epoch):
    lr = max(init_lr * (0.9 ** (epoch // 10)), 0.0002)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / Y_pred.size, w


def scalar_loss(data, loss_recon, mu_z, logvar_z, beta):
    # calculate scalar loss of VAE1
    loss_recon *= data.size(1) * data.size(2) * data.size(3)
    loss_recon.div(data.size(0))
    loss_kl = kl_divergence(mu_z, logvar_z)
    return loss_recon + beta * loss_kl, loss_kl


#########################################
##### train vaDE
#########################################
def pretrain_2stageVAEmodel_SSIM(model, dataloader, pre_epoch=30, save_path='', device='cpu'):
    print(os.path.join(save_path, 'pretrain_model.pk'))
    if os.path.exists(os.path.join(save_path, 'pretrain_model.pk')):
        model.load_state_dict(torch.load(os.path.join(save_path, 'pretrain_model.pk')))
        print("pretrained-weight loaded!!!")
        return

    opti = optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.mu_logvar_gen_1.parameters()},
        {'params': model.mu_logvar_gen_2.parameters()},
        {'params': model.decoder_1.parameters()},
        {'params': model.decoder_2.parameters()},
    ],
        lr=1e-3, betas=(0.9, 0.999))
    model.to(device)

    print(f'Pretraining......on {device}')
    epoch_bar = tqdm(range(pre_epoch))
    for _ in epoch_bar:
        L = ssim = l1_loss = 0
        for x, y in dataloader:
            x = x.to(device)

            x_recon_hi, _, _, z_1, x_recon_lo, _, _, z_2 = model(x)

            x_recon_hi = torch.sigmoid(x_recon_hi)
            x_recon_lo = torch.sigmoid(x_recon_lo)

            msssim_similarity = msssim(x, x_recon_hi, normalize='relu') + msssim(x, x_recon_lo, normalize='relu')
            l1 = F.l1_loss(x_recon_hi, x, reduction='sum').div(x.size(0)) + F.l1_loss(x_recon_lo, x,
                                                                                      reduction='sum').div(x.size(0))
            loss = (2 - msssim_similarity) + l1

            ssim += msssim_similarity.item()
            l1_loss += l1.item()

            L += loss.detach().cpu().numpy()

            opti.zero_grad()
            loss.backward()
            opti.step()

        epoch_bar.write('\nL={:.4f} ssim={:.4f} L1={:.4f}\n'.format(L / len(dataloader), ssim / len(dataloader),
                                                                    l1_loss / len(dataloader)))
    # reset to save weight???? TODO: check
    # model.logvar_l.load_state_dict(model.mu_l.state_dict())
    torch.save(model.state_dict(), os.path.join(save_path, 'pretrain_model.pk'))


def pretrain_2stageVaDE_model_SSIM(model, dataloader, pre_epoch=30, save_path='', device='cpu'):
    if os.path.exists(os.path.join(save_path, 'pretrain_model.pk')):
        print(os.path.join(save_path, 'pretrain_model.pk'))
        model.load_state_dict(torch.load(os.path.join(save_path, 'pretrain_model.pk')))
        return

    opti = optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.mu_logvar_gen_1.parameters()},
        {'params': model.mu_logvar_gen_2.parameters()},
        {'params': model.decoder_1.parameters()},
        {'params': model.decoder_2.parameters()},
    ],
        lr=5e-4, betas=(0.9, 0.999))

    model.to(device)
    print(f'Pretraining......on {device}')
    epoch_bar = tqdm(range(pre_epoch))
    for _ in epoch_bar:
        L = ssim = l1_loss = 0
        for x, y in dataloader:
            x = x.to(device)

            x_recon_hi, _, _, z_1, x_recon_lo, _, _, z_2 = model(x)

            x_recon_hi = torch.sigmoid(x_recon_hi)
            x_recon_lo = torch.sigmoid(x_recon_lo)

            msssim_similarity = msssim(x, x_recon_hi, normalize='relu') + msssim(x, x_recon_lo, normalize='relu')
            l1 = F.l1_loss(x_recon_hi, x, reduction='sum').div(x.size(0)) + F.l1_loss(x_recon_lo, x,
                                                                                      reduction='sum').div(x.size(0))
            loss = (2 - msssim_similarity) + l1

            ssim += msssim_similarity.item()
            l1_loss += l1.item()

            L += loss.detach().cpu().numpy()

            opti.zero_grad()
            loss.backward()
            opti.step()

        epoch_bar.write('\nL={:.4f} ssim={:.4f} L1={:.4f}\n'.format(L / len(dataloader), ssim / len(dataloader),
                                                                    l1_loss / len(dataloader)))
    # reset to save weight???? TODO: check
    # model.logvar_l.load_state_dict(model.mu_l.state_dict())

    ### validate pretrained model
    Z, Y = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)

            _, _, _, _, _, _, _, z_2 = model(x)
            # assert F.mse_loss(mu_z, logvar_z) == 0
            Z.append(z_2)
            Y.append(y)

    torch.save(model.state_dict(), os.path.join(save_path, 'pretrain_model.pk'))
    # convert to tensor
    Z = torch.cat(Z, 0).detach().cpu().numpy()
    Y = torch.cat(Y, 0).detach().numpy()

    # select number of clusters
    n_components = np.arange(1, 12)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(Z) for n in n_components]
    bic = [m.bic(Z) for m in models]
    aic = [m.aic(Z) for m in models]
    plt.plot(n_components, bic, label='BIC')
    plt.plot(n_components, aic, label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.savefig(os.path.join(save_path, 'BIC_AIC.png'))

    gmm = GaussianMixture(n_components=model.ydim, covariance_type='diag')
    pre = gmm.fit_predict(Z)
    print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

    model.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
    model.mu_c.data = torch.from_numpy(gmm.means_.T).cuda().float()
    model.log_sigma2_c.data = torch.from_numpy(gmm.covariances_.T).cuda().float()

    torch.save(model.state_dict(), os.path.join(save_path, 'pretrain_model.pk'))


def pretrain_EnhancedVAE_model_SSIM(model, dataloader, pre_epoch=30, save_path='', device='cpu'):
    print(os.path.join(save_path, 'pretrain_model.pk'))
    if os.path.exists(os.path.join(save_path, 'pretrain_model.pk')):
        model.load_state_dict(torch.load(os.path.join(save_path, 'pretrain_model.pk')))
        print("loaded!!!")
        return

    opti = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    model.to(device)

    print(f'Pretraining......on {device}')
    epoch_bar = tqdm(range(pre_epoch))
    for _ in epoch_bar:
        L = ssim = l1_loss = 0
        for x, _ in dataloader:
            x = x.to(device)

            x_recon, mu_z, logvar_z, z = model(x)
            x_recon = torch.sigmoid(x_recon)

            # loss_recon = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(x.size(0))
            msssim_similarity = msssim(x, x_recon, normalize='relu')
            l1 = F.l1_loss(x_recon, x, reduction='sum').div(x.size(0))  # F.mse_loss(x_recon, x)
            loss = (1 - msssim_similarity) + l1  # 0.00005 * l1

            ssim += msssim_similarity.item()
            l1_loss += l1.item()

            L += loss.detach().cpu().numpy()

            opti.zero_grad()
            loss.backward()
            opti.step()

        epoch_bar.write('\nL={:.4f} ssim={:.4f} L1={:.4f}\n'.format(L / len(dataloader), ssim / len(dataloader),
                                                                    l1_loss / len(dataloader)))
    # reset to save weight???? TODO: check
    # model.logvar_l.load_state_dict(model.mu_l.state_dict())
    torch.save(model.state_dict(), os.path.join(save_path, 'pretrain_model.pk'))


def pretrain_vaDE_model_SSIM(model, dataloader, pre_epoch=30, save_path='', device='cpu'):
    print(os.path.join(save_path, 'pretrain_model.pk'))
    if os.path.exists(os.path.join(save_path, 'pretrain_model.pk')):
        model.load_state_dict(torch.load(os.path.join(save_path, 'pretrain_model.pk')))
        return

    opti = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    model.to(device)

    print(f'Pretraining......on {device}')
    epoch_bar = tqdm(range(pre_epoch))
    for _ in epoch_bar:
        L = ssim = l1_loss = 0
        for x, y in dataloader:
            x = x.to(device)

            mu_logvar = model.encode(x)
            z = model.mu_l(mu_logvar)
            x_recon = model.decode(z)

            x_recon = torch.sigmoid(x_recon)
            msssim_similarity = msssim(x, x_recon, normalize='relu')
            l1 = F.l1_loss(x_recon, x, reduction='sum').div(x.size(0))
            loss = (1 - msssim_similarity) + l1

            ssim += msssim_similarity.item()
            l1_loss += l1.item()

            L += loss.detach().cpu().numpy()

            opti.zero_grad()
            loss.backward()
            opti.step()

        epoch_bar.write('\nL={:.4f} ssim={:.4f} L1={:.4f}\n'.format(L / len(dataloader), ssim / len(dataloader),
                                                                    l1_loss / len(dataloader)))
    # reset to save weight???? TODO: check
    model.logvar_l.load_state_dict(model.mu_l.state_dict())

    ### validate pretrained model
    Z, Y = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)

            mu_logvar = model.encode(x)
            z = model.mu_l(mu_logvar)
            # assert F.mse_loss(mu_z, logvar_z) == 0
            Z.append(z)
            Y.append(y)

    torch.save(model.state_dict(), os.path.join(save_path, 'pretrain_model.pk'))
    # convert to tensor
    Z = torch.cat(Z, 0).detach().cpu().numpy()
    Y = torch.cat(Y, 0).detach().numpy()

    # select number of clusters
    n_components = np.arange(1, 12)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(Z) for n in n_components]
    bic = [m.bic(Z) for m in models]
    aic = [m.aic(Z) for m in models]
    plt.plot(n_components, bic, label='BIC')
    plt.plot(n_components, aic, label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.savefig(os.path.join(save_path, 'BIC_AIC.png'))

    gmm = GaussianMixture(n_components=model.ydim, covariance_type='diag')
    pre = gmm.fit_predict(Z)
    print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

    model.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
    model.mu_c.data = torch.from_numpy(gmm.means_.T).cuda().float()
    model.log_sigma2_c.data = torch.from_numpy(gmm.covariances_.T).cuda().float()

    torch.save(model.state_dict(), os.path.join(save_path, 'pretrain_model.pk'))


def pretrain_vaDE_model(model, dataloader, pre_epoch=30, save_path='', device='cpu'):
    print(os.path.join(save_path, 'pretrain_model.pk'))
    if os.path.exists(os.path.join(save_path, 'pretrain_model.pk')):
        model.load_state_dict(torch.load(os.path.join(save_path, 'pretrain_model.pk')))
        return

    Loss = nn.MSELoss()
    opti = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    model.to(device)

    print(f'Pretraining......on {device}')
    epoch_bar = tqdm(range(pre_epoch))
    for _ in epoch_bar:
        L = 0
        for x, y in dataloader:
            x = x.to(device)

            mu_logvar = model.encode(x)
            z = model.mu_l(mu_logvar)
            x_recon = model.decode(z)

            loss = Loss(x, torch.sigmoid(x_recon))

            L += loss.detach().cpu().numpy()

            opti.zero_grad()
            loss.backward()
            opti.step()

        epoch_bar.write('\n L2={:.4f}\n'.format(L / len(dataloader)))
    # reset to save weight???? TODO: check
    model.logvar_l.load_state_dict(model.mu_l.state_dict())

    ### validate pretrained model
    Z, Y = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)

            mu_logvar = model.encode(x)
            z = model.mu_l(mu_logvar)
            # assert F.mse_loss(mu_z, logvar_z) == 0
            Z.append(z)
            Y.append(y)

    # convert to tensor
    Z = torch.cat(Z, 0).detach().cpu().numpy()
    Y = torch.cat(Y, 0).detach().numpy()

    # select number of clusters
    n_components = np.arange(1, 12)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(Z) for n in n_components]
    bic = [m.bic(Z) for m in models]
    aic = [m.aic(Z) for m in models]
    plt.plot(n_components, bic, label='BIC')
    plt.plot(n_components, aic, label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.savefig(os.path.join(save_path, 'BIC_AIC.png'))

    gmm = GaussianMixture(n_components=model.ydim, covariance_type='diag')
    pre = gmm.fit_predict(Z)
    print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

    model.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
    model.mu_c.data = torch.from_numpy(gmm.means_.T).cuda().float()
    model.log_sigma2_c.data = torch.from_numpy(gmm.covariances_.T).cuda().float()

    torch.save(model.state_dict(), os.path.join(save_path, 'pretrain_model.pk'))


#########################################
##### 2 stage VAE training ##############
#########################################
def train_2stage_VAE_epoch(num_epochs, VAE_1, VAE_2, optimizer_1, optimizer_2, train_loader, gamma=0.8, device='cuda'):
    """
        Train a VAE model with standard ELBO objective function for one single epoch

        Params :
            epoch (int) : the considered epoch
            model (nn.Module) : VAE model to train
            optimizer (optim.Optimizer) : Optimizer used for training
            train_loader (DataLoader) : Dataloader used for training

        Return the average loss, as well as average of the two main terms (reconstruction and KL)
    """
    # toggle model to train mode
    VAE_1.train()
    VAE_2.train()

    # Store the different loss per iteration (=batch)
    loss_overall_iter = []
    global_VAE_iter_1, kl_loss_iter_1, recon_loss_iter_1 = [], [], []
    global_VAE_iter_2, kl_loss_iter_2, recon_loss_iter_2 = [], [], []

    start = timer()

    criterion_recon = nn.BCEWithLogitsLoss().to(device)  # more stable than handmade sigmoid as last layer and BCELoss

    # each `data` is of BATCH_SIZE samples and has shape [batch_size, 4, 128, 128]
    for batch_idx, (data, label) in enumerate(train_loader):
        data = Variable(data).to(device)
        # tensor m*c*n*n
        # label tensor m*1

        ### encode conditions
        if VAE_1.conditional:
            # label_channel = torch.reshape(label, [len(data), 1, 1, 1])
            # ones = torch.ones((len(data), 1, data.shape[2], data.shape[2]))
            # # (m*1*1*1) * (m*1*64*64)
            # label_channel = Variable(label_channel * ones).to(device)
            # # concatenate to additional channel
            # data_condition = torch.cat([data, label_channel], axis=1)
            # data = data_condition
            # push whole batch of data through VAE.forward() to get recon_loss

            y_onehot = torch.zeros((len(label), 7))
            y_onehot[torch.arange(len(label)), label] = 1
            label = Variable(y_onehot.float()).to(device)
            x_recon_1, mu_z_1, logvar_z_1, _ = VAE_1((data, label))
            x_recon_2, mu_z_2, logvar_z_2, _ = VAE_2((data, label))
        else:
            x_recon_1, mu_z_1, logvar_z_1, _ = VAE_1(data)
            x_recon_2, mu_z_2, logvar_z_2, _ = VAE_2(data)

        # calculate scalar loss of VAE1
        loss_recon_1 = criterion_recon(x_recon_1, data)

        loss_VAE_1, loss_kl_1 = scalar_loss(data, loss_recon_1, mu_z_1, logvar_z_1, VAE_1.beta)
        # calculate scalar loss of VAE2
        loss_recon_2 = criterion_recon(x_recon_2, data)
        loss_VAE_2, loss_kl_2 = scalar_loss(data, loss_recon_2, mu_z_2, logvar_z_2, VAE_2.beta)
        # total loss
        loss_overall = loss_VAE_2 + gamma * loss_VAE_1  # as auxiliary loss

        optimizer_1.zero_grad();
        optimizer_2.zero_grad()
        loss_overall.backward()
        optimizer_1.step();
        optimizer_2.step()

        # record the loss
        # record the loss
        loss_overall_iter.append(loss_overall.item())
        global_VAE_iter_1.append(loss_VAE_1.item());
        global_VAE_iter_2.append(loss_VAE_2.item())
        recon_loss_iter_1.append(loss_recon_1.item());
        recon_loss_iter_2.append(loss_recon_2.item())
        kl_loss_iter_1.append(loss_kl_1.item());
        kl_loss_iter_2.append(loss_kl_2.item())

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tVAE1 Loss: {:.6f}\tVAE2 Loss: {:.6f}'.format(
                num_epochs, batch_idx * len(data), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                loss_VAE_1.item(), loss_VAE_2.item()), end='\r')

    if (num_epochs % 5 == 0) or (num_epochs == 1):
        print('==========> Epoch: {} ==========> Average loss: {:.4f}'.format(num_epochs, np.mean(loss_overall_iter)))
        print(f'{timer() - start:.2f} seconds elapsed in epoch.')
        print(f'VAE_1 VAE loss: {np.mean(global_VAE_iter_1):.2f}, KL loss : {np.mean(global_VAE_iter_2):.2f}')
        print(f'VAE_1 reconstruction loss: {np.mean(recon_loss_iter_1):.2f}, KL loss : {np.mean(kl_loss_iter_1):.2f}')
        print(f'VAE_2 reconstruction loss: {np.mean(recon_loss_iter_2):.2f}, KL loss : {np.mean(kl_loss_iter_2):.2f}')

    return np.mean(loss_overall_iter), np.mean(kl_loss_iter_1), np.mean(kl_loss_iter_2), np.mean(
        recon_loss_iter_1), np.mean(recon_loss_iter_2)


def test_2stage_VAE_epoch(num_epochs, VAE_1, VAE_2, test_loader, gamma=0.8, device='cuda'):
    """
        Train a VAE model with standard ELBO objective function for one single epoch

        Params :
            epoch (int) : the considered epoch
            model (nn.Module) : VAE model to train
            optimizer (optim.Optimizer) : Optimizer used for training
            train_loader (DataLoader) : Dataloader used for training

        Return the average loss, as well as average of the two main terms (reconstruction and KL)
    """
    with torch.no_grad():
        VAE_1.eval();
        VAE_2.eval()
        # Store the different loss per iteration (=batch)
        loss_overall_iter = []
        global_VAE_iter_1, kl_loss_iter_1, recon_loss_iter_1 = [], [], []
        global_VAE_iter_2, kl_loss_iter_2, recon_loss_iter_2 = [], [], []

        criterion_recon = nn.BCEWithLogitsLoss().to(
            device)  # more stable than handmade sigmoid as last layer and BCELoss

        # each `data` is of BATCH_SIZE samples and has shape [batch_size, 4, 128, 128]
        for batch_idx, (data, label) in enumerate(test_loader):
            data = Variable(data).to(device)

            ### encode conditions
            if VAE_1.conditional:
                # label_channel = torch.reshape(label, [len(data), 1, 1, 1])
                # ones = torch.ones((len(data), 1, data.shape[2], data.shape[2]))
                # # (m*1*1*1) * (m*1*64*64)
                # label_channel = Variable(label_channel * ones).to(device)
                # # concatenate to additional channel
                # data_condition = torch.cat([data, label_channel], axis=1)
                # data = data_condition
                # # push whole batch of data through VAE.forward() to get recon_loss
                # label = Variable(label.float()).to(device)
                # x_recon_1, mu_z_1, logvar_z_1, _ = VAE_1((data_condition, label))
                # x_recon_2, mu_z_2, logvar_z_2, _ = VAE_2((data_condition, label))

                y_onehot = torch.zeros((len(label), 7))
                y_onehot[torch.arange(len(label)), label] = 1
                label = Variable(y_onehot.float()).to(device)
                x_recon_1, mu_z_1, logvar_z_1, _ = VAE_1((data, label))
                x_recon_2, mu_z_2, logvar_z_2, _ = VAE_2((data, label))
            else:
                x_recon_1, mu_z_1, logvar_z_1, _ = VAE_1(data)
                x_recon_2, mu_z_2, logvar_z_2, _ = VAE_2(data)

            # calculate scalar loss of VAE1
            loss_recon_1 = criterion_recon(x_recon_1, data)
            loss_VAE_1, loss_kl_1 = scalar_loss(data, loss_recon_1, mu_z_1, logvar_z_1, VAE_1.beta)
            # calculate scalar loss of VAE2
            loss_recon_2 = criterion_recon(x_recon_2, data)
            loss_VAE_2, loss_kl_2 = scalar_loss(data, loss_recon_2, mu_z_2, logvar_z_2, VAE_2.beta)
            # total loss
            loss_overall = loss_VAE_2 + gamma * loss_VAE_1  # as auxiliary loss

            # record the loss
            loss_overall_iter.append(loss_overall.item())
            global_VAE_iter_1.append(loss_VAE_1.item());
            global_VAE_iter_2.append(loss_VAE_2.item())
            recon_loss_iter_1.append(loss_recon_1.item());
            recon_loss_iter_2.append(loss_recon_2.item())
            kl_loss_iter_1.append(loss_kl_1.item());
            kl_loss_iter_2.append(loss_kl_2.item())

    if (num_epochs % 10 == 0) or (num_epochs == 1):
        print('--------> Test Error for Epoch: {} --------> Average loss: {:.4f}'.format(num_epochs,
                                                                                         np.mean(loss_overall_iter)))
    return np.mean(loss_overall_iter), np.mean(kl_loss_iter_1), np.mean(kl_loss_iter_2), np.mean(
        recon_loss_iter_1), np.mean(recon_loss_iter_2)


def train_2stage_VAE_model(num_epochs, VAE_1, VAE_2, optimizer1, optimizer2, train_loader, valid_loader,
                           gamma=1, save_path='', device='cpu'):
    """
        Define VAE 1:

        Params :
            epochs (int) : Maximum number of epochs
            model (nn.Module) : VAE model to train
            optimizer (optim.Optimizer) : Optimizer used for training
            train_loader (DataLoader) : Dataloader used for training
            test_loader (DataLoader) : Dataloader used for evaluation
            saving_path (string) : path to the folder to store the best model

        Return a DataFrame containing the training history, as well as the trained model and the best epoch
    """
    # Number of epochs already trained (if pre trained)
    try:
        print(f'VAE1, VAE2 has been trained for: {VAE_1.epochs}, {VAE_2.epochs} epochs.\n')
    except:
        VAE_1.epochs, VAE_2.epochs = 0, 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()
    best_epoch = 0

    history = []
    early_stopping = EarlyStopping(patience=20, verbose=True, path=save_path)
    lr_schedul_VAE_1 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer1, step_size=40, gamma=0.6)
    lr_schedul_VAE_2 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer2, step_size=40, gamma=0.6)

    for epoch in range(num_epochs):
        VAE_loss, kl_1, kl_2, recon_1, recon_2 = train_2stage_VAE_epoch(epoch, VAE_1, VAE_2,
                                                                        optimizer1,
                                                                        optimizer2,
                                                                        train_loader,
                                                                        gamma, device)
        VAE_loss_val, kl_val_1, kl_val_2, recon_val_1, recon_val_2 = test_2stage_VAE_epoch(epoch, VAE_1, VAE_2,
                                                                                           valid_loader, gamma, device)

        # early stopping takes the validation loss to check if it has decereased,
        # if so, model is saved, if not for 'patience' time in a row, the training loop is broken
        early_stopping(VAE_loss_val, VAE_1.epochs)

        history.append(
            [VAE_loss, kl_1, kl_2, recon_1, recon_2, VAE_loss_val, kl_val_1, kl_val_2, recon_val_1, recon_val_2])
        VAE_1.epochs, VAE_2.epochs = VAE_1.epochs + 1, VAE_2.epochs + 1

        lr_schedul_VAE_1.step();
        lr_schedul_VAE_2.step()

        if early_stopping.early_stop:
            print(f'#### Early stopping occured. Best model saved is from epoch {early_stopping.stop_epoch}')
            break

    ### SAVE the model
    early_stopping.save_model((VAE_1, VAE_2))
    best_epoch = early_stopping.stop_epoch

    history = pd.DataFrame(
        history,
        columns=['VAE_loss', 'kl_1', 'kl_2', 'recon_1', 'recon_2', 'VAE_loss_val', 'kl_val_1', 'kl_val_2',
                 'recon_val_1', 'recon_val_2']
    )

    # time
    total_time = timer() - overall_start
    print(f'{total_time:.2f} total seconds elapsed. {total_time / (num_epochs):.2f} seconds per epoch.')
    print('######### TRAINING FINISHED ##########')

    # Attach the optimizer
    VAE_1.optimizer = optimizer1;
    VAE_2.optimizer = optimizer2

    return VAE_1, VAE_2, history, best_epoch


###################################################
##### 2 stage InfoMaxVAE ##############
###################################################
def train_2stage_infoMaxVAE_epoch(num_epochs, VAE_1, VAE_2, optim_VAE1, optim_VAE2, MLP_1, MLP_2, opti_MLP1, opti_MLP2,
                                  train_loader, double_embed=False, gamma=1, device='cuda'):
    """
        Train a VAE model with standard ELBO objective function for one single epoch

        Params :
            epoch (int) : the considered epoch
            model (nn.Module) : VAE model to train
            optimizer (optim.Optimizer) : Optimizer used for training
            train_loader (DataLoader) : Dataloader used for training

        Return the average loss, as well as average of the two main terms (reconstruction and KL)
    """
    # toggle model to train mode
    VAE_1.train()
    VAE_2.train()

    # Store the different loss per iteration (=batch)
    loss_overall_iter = []
    global_VAE_iter_1, kl_loss_iter_1, recon_loss_iter_1, MI_iter_1 = [], [], [], []
    global_VAE_iter_2, kl_loss_iter_2, recon_loss_iter_2, MI_iter_2 = [], [], [], []

    start = timer()

    criterion_recon = nn.BCEWithLogitsLoss().to(device)  # more stable than handmade sigmoid as last layer and BCELoss

    # each `data` is of BATCH_SIZE samples and has shape [batch_size, 4, 128, 128]
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data).to(device)

        # push whole batch of data through VAE.forward() to get recon_loss
        x_recon_1, mu_z_1, logvar_z_1, z_1 = VAE_1(data)
        scores_1 = MLP_1(data, z_1)

        if not double_embed:
            x_recon_2, mu_z_2, logvar_z_2, z_2 = VAE_2(data)
            scores_2 = MLP_2(data, z_2)
        else:
            # double embedding pass Z_1 to VAE_2
            x_recon_2, mu_z_2, logvar_z_2, z_2 = VAE_2(z_1)
            scores_2 = MLP_2(z_1, z_2)

        # Estimation of the Mutual Info between X and Z
        MI_xz_1 = infoNCE_bound(scores_1)
        MI_xz_2 = infoNCE_bound(scores_2)

        # calculate scalar loss of VAE1
        loss_recon_1 = criterion_recon(x_recon_1, data)
        loss_VAE_1, loss_kl_1 = scalar_loss(data, loss_recon_1, mu_z_1, logvar_z_1, VAE_1.beta)
        # calculate scalar loss of VAE2
        loss_recon_2 = criterion_recon(x_recon_2, data)
        loss_VAE_2, loss_kl_2 = scalar_loss(data, loss_recon_2, mu_z_2, logvar_z_2, VAE_2.beta)
        # total loss
        # loss_overall = loss_VAE_1 + gamma * loss_VAE_2
        loss_overall = (loss_VAE_2 - VAE_2.alpha * MI_xz_2) + gamma * (loss_VAE_1 - VAE_1.alpha * MI_xz_1)

        # Step 1 : Optimization of VAE based on the current MI estimation
        optim_VAE1.zero_grad();
        optim_VAE2.zero_grad()
        loss_overall.backward(retain_graph=True)  # Important argument, we backpropagated two times over MI_xz)
        optim_VAE1.step();
        optim_VAE2.step()

        # Step 2 : Optimization of the MLP to improve the MI estimation
        opti_MLP1.zero_grad();
        opti_MLP2.zero_grad()
        MI_loss_1 = -MI_xz_1
        MI_loss_2 = -MI_xz_2
        MI_loss_1.backward(retain_graph=True)  # Important argument
        MI_loss_2.backward()
        opti_MLP1.step();
        opti_MLP2.step()

        # record the loss
        loss_overall_iter.append(loss_overall.item())
        global_VAE_iter_1, global_VAE_iter_2 = global_VAE_iter_1 + [loss_VAE_1.item()], global_VAE_iter_2 + [
            loss_VAE_2.item()]
        recon_loss_iter_1, recon_loss_iter_2 = recon_loss_iter_1 + [loss_recon_1.item()], global_VAE_iter_2 + [
            loss_recon_2.item()]
        kl_loss_iter_1, kl_loss_iter_2 = kl_loss_iter_1 + [loss_kl_1.item()], kl_loss_iter_2 + [loss_kl_2.item()]

        MI_iter_1.append(MI_xz_1.item());
        MI_iter_2.append(MI_xz_2.item())

    if (num_epochs % 10 == 0) or (num_epochs == 1):
        print('==========> Epoch: {} ==========> Average loss: {:.4f}'.format(num_epochs, np.mean(loss_overall_iter)))
        print(f'{timer() - start:.2f} seconds elapsed in epoch.')
        print(f'VAE_1 VAE loss: {np.mean(global_VAE_iter_1):.2f}, KL loss : {np.mean(global_VAE_iter_2):.2f}')
        print(
            f'VAE_1 reconstruction loss: {np.mean(recon_loss_iter_1):.2f}, KL loss : {np.mean(kl_loss_iter_1):.2f} MI : {np.mean(MI_iter_1):.2f}')
        print(
            f'VAE_2 reconstruction loss: {np.mean(recon_loss_iter_2):.2f}, KL loss : {np.mean(kl_loss_iter_2):.2f} MI : {np.mean(MI_iter_2):.2f}')

    return np.mean(loss_overall_iter), np.mean(kl_loss_iter_1), np.mean(kl_loss_iter_2), np.mean(
        recon_loss_iter_1), np.mean(recon_loss_iter_2), np.mean(MI_iter_1), np.mean(MI_iter_2)


def test_2stage_infoMaxVAE_epoch(num_epochs, VAE_1, VAE_2, optim_VAE1, optim_VAE2, MLP_1, MLP_2, opti_MLP1, opti_MLP2,
                                 test_loader, double_embed=False, gamma=0.8, device='cuda'):
    """
        Train a VAE model with standard ELBO objective function for one single epoch
        TODO: Complete Documentation
        Params :
            epoch (int) : the considered epoch
            model (nn.Module) : VAE model to train
            optimizer (optim.Optimizer) : Optimizer used for training
            train_loader (DataLoader) : Dataloader used for training
    """
    with torch.no_grad():
        VAE_1.eval()
        VAE_2.eval()
        MLP_1.eval()
        MLP_2.eval()

        # Store the different loss per iteration (=batch)
        loss_overall_iter = []
        global_VAE_iter_1, kl_loss_iter_1, recon_loss_iter_1, MI_iter_1 = [], [], [], []
        global_VAE_iter_2, kl_loss_iter_2, recon_loss_iter_2, MI_iter_2 = [], [], [], []

        criterion_recon = nn.BCEWithLogitsLoss().to(
            device)  # more stable than handmade sigmoid as last layer and BCELoss

        # each `data` is of BATCH_SIZE samples and has shape [batch_size, 4, 128, 128]
        for batch_idx, (data, _) in enumerate(test_loader):
            data = Variable(data).to(device)

            # push whole batch of data through VAE.forward() to get recon_loss
            x_recon_1, mu_z_1, logvar_z_1, z_1 = VAE_1(data)
            scores_1 = MLP_1(data, z_1)

            if not double_embed:
                x_recon_2, mu_z_2, logvar_z_2, z_2 = VAE_2(data)
                scores_2 = MLP_2(data, z_2)
            else:
                # double embedding pass Z_1 to VAE_2
                x_recon_2, mu_z_2, logvar_z_2, z_2 = VAE_2(z_1)
                scores_2 = MLP_2(z_1, z_2)

            # Estimation of the Mutual Info between X and Z
            MI_xz_1 = infoNCE_bound(scores_1)
            MI_xz_2 = infoNCE_bound(scores_2)

            # calculate scalar loss of VAE1
            loss_recon_1 = criterion_recon(x_recon_1, data)
            loss_VAE_1, loss_kl_1 = scalar_loss(data, loss_recon_1, mu_z_1, logvar_z_1, VAE_1.beta)
            # calculate scalar loss of VAE2
            loss_recon_2 = criterion_recon(x_recon_2, data)
            loss_VAE_2, loss_kl_2 = scalar_loss(data, loss_recon_2, mu_z_2, logvar_z_2, VAE_2.beta)
            # total loss
            # loss_overall = loss_VAE_1 + gamma * loss_VAE_2
            loss_overall = (loss_VAE_2 - VAE_2.alpha * MI_xz_2) + gamma * (loss_VAE_1 - VAE_1.alpha * MI_xz_1)

            # record the loss
            loss_overall_iter.append(loss_overall.item())
            global_VAE_iter_1, global_VAE_iter_2 = global_VAE_iter_1 + [loss_VAE_1.item()], global_VAE_iter_2 + [
                loss_VAE_2.item()]
            recon_loss_iter_1, recon_loss_iter_2 = recon_loss_iter_1 + [loss_recon_1.item()], global_VAE_iter_2 + [
                loss_recon_2.item()]
            kl_loss_iter_1, kl_loss_iter_2 = kl_loss_iter_1 + [loss_kl_1.item()], kl_loss_iter_2 + [loss_kl_2.item()]
            MI_iter_1.append(MI_xz_1.item())
            MI_iter_2.append(MI_xz_2.item())
        if num_epochs % 10 == 0:
            print(f'--------> Test Error for Epoch: {num_epochs} --------> Average loss: {np.mean(loss_overall_iter)}')

        return np.mean(loss_overall_iter), np.mean(kl_loss_iter_1), np.mean(kl_loss_iter_2), np.mean(
            recon_loss_iter_1), np.mean(recon_loss_iter_2), np.mean(MI_iter_1), np.mean(MI_iter_2)


def train_2stage_infoMaxVAE_model(num_epochs, VAE_1, VAE_2, opti_VAE1, opti_VAE2, MLP_1, MLP_2, opti_MLP1, opti_MLP2,
                                  train_loader, valid_loader, gamma=0.8,
                                  save_path='', double_embed=False, device='cuda'):
    # Number of epochs already trained (if pre trained)
    try:
        print(f'VAE1, VAE2 has been trained for: {VAE_1.epochs}, {VAE_2.epochs} epochs.\n')
    except:
        VAE_1.epochs, VAE_2.epochs = 0, 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()
    best_epoch = 0

    history = []
    early_stopping = EarlyStopping(patience=30, verbose=True, path=save_path)
    lr_schedul_VAE_1 = torch.optim.lr_scheduler.StepLR(optimizer=opti_VAE1, step_size=40, gamma=0.6)
    lr_schedul_VAE_2 = torch.optim.lr_scheduler.StepLR(optimizer=opti_VAE2, step_size=40, gamma=0.6)
    lr_schedul_MLP1 = torch.optim.lr_scheduler.StepLR(optimizer=opti_MLP1, step_size=40, gamma=0.6)
    lr_schedul_MLP2 = torch.optim.lr_scheduler.StepLR(optimizer=opti_MLP2, step_size=40, gamma=0.6)

    for epoch in range(num_epochs):
        VAE_loss, kl_1, kl_2, recon_1, recon_2, MI_1, MI_2 = train_2stage_infoMaxVAE_epoch(
            epoch, VAE_1, VAE_2,
            opti_VAE1,
            opti_VAE2, MLP_1, MLP_2, opti_MLP1, opti_MLP2,
            train_loader, double_embed,
            gamma,
            device)
        VAE_loss_val, kl_val_1, kl_val_2, recon_val_1, recon_val_2, MI_val_1, MI_val_2 = test_2stage_infoMaxVAE_epoch(
            epoch, VAE_1, VAE_2,
            opti_VAE1,
            opti_VAE2, MLP_1, MLP_2, opti_MLP1, opti_MLP2,
            valid_loader, double_embed,
            gamma,
            device)

        # early stopping takes the validation loss to check if it has decereased,
        # if so, model is saved, if not for 'patience' time in a row, the training loop is broken
        early_stopping(VAE_loss_val, VAE_1.epochs)

        history.append(
            [VAE_loss, kl_1, kl_2, recon_1, recon_2, MI_1, MI_2, VAE_loss_val, kl_val_1, kl_val_2, recon_val_1,
             recon_val_2,
             MI_val_1, MI_val_2])
        # storing model epochs
        VAE_1.epochs, VAE_2.epochs, MLP_1.epochs, MLP_2.epochs = VAE_1.epochs + 1, VAE_2.epochs + 1, MLP_1.epochs + 1, MLP_2.epochs + 1

        lr_schedul_VAE_1.step()
        lr_schedul_VAE_2.step()
        lr_schedul_MLP1.step()
        lr_schedul_MLP2.step()

        if early_stopping.early_stop:
            print(f'#### Early stopping occured. Best model saved is from epoch {early_stopping.stop_epoch}')
            break

    ### SAVE the model
    early_stopping.save_model((VAE_1, MLP_1, VAE_2, MLP_2))
    best_epoch = early_stopping.stop_epoch

    history = pd.DataFrame(
        history,
        columns=['VAE_loss', 'kl_1', 'kl_2', 'recon_1', 'recon_2', 'MI_1', 'MI_2', 'VAE_loss_val', 'kl_val_1',
                 'kl_val_2',
                 'recon_val_1', 'recon_val_2', 'MI_val_1', 'MI_val_2']
    )

    # time
    total_time = timer() - overall_start
    print(f'{total_time:.2f} total seconds elapsed. {total_time / (num_epochs):.2f} seconds per epoch.')
    print('######### TRAINING FINISHED ##########')

    # Attach the optimizer
    VAE_1.optimizer = opti_VAE1
    VAE_2.optimizer = opti_VAE2
    MLP_1.optimizer = opti_MLP1
    MLP_2.optimizer = opti_MLP2

    return VAE_1, VAE_2, MLP_1, MLP_2, history, best_epoch


###################################################
##### Vanilla VAE and SCVAE training ##############
###################################################
class VAEXperiment(pl.LightningModule):
    def __init__(self,
                 model: AbstractModel,
                 params: dict, log_path: str = '') -> None:
        super(VAEXperiment, self).__init__()

        self.model = model
        self.params = params
        self.curr_device = None
        self.log_path = log_path
        # create log path
        make_path(log_path)

    def parameter_histogram(self):
        for n, p in self.model.state_dict().items():  # .named_parameters():
            self.logger.experiment.add_histogram(n, p, self.global_step)

    def plot_computation_graph(self):
        sample_img = torch.rand(1, self.model.input_channels, self.model.input_size, self.model.input_size).to(
            self.curr_device)
        self.logger.experiment.add_graph(self.model, sample_img)

    '''
        extracted method for stepping image data with model
    '''

    def training_step(self, batch, batch_idx, optimizer_idx=0) -> dict:
        # self.log_dict({key: val.item() for key, val in train_loss.items()}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # train_loss['log'] = {key: val.item() for key, val in train_loss.items()}
        img, labels = batch
        self.curr_device = img.device
        return self.model.training_step(batch, batch_idx, optimizer_idx)

    def backward(self, loss: Tensor, optimizer: Optimizer, optimizer_idx: int, *args, **kwargs) -> None:
        self.model.backward(loss, optimizer, optimizer_idx)

    def on_after_backward(self) -> None:
        if self.global_step % 200 == 0:
            self.parameter_histogram()

    def training_epoch_end(self, outputs: list):
        """
        :param outputs: [{loss_name, val}]
        """
        # map reduce the sum of each loss_name, values must be positive
        outputs = self.model.history if len(self.model.history) != 0 else outputs
        # sum_loss = dict(functools.reduce(operator.add, map(collections.Counter, outputs)))
        sum_loss = pd.DataFrame.from_records(outputs).mean().to_dict()
        for k, v in sum_loss.items():
            self.logger.experiment.add_scalar(k, v, self.current_epoch)
        ### reset record history
        self.model.history = []
        ### plot dynamic computation graph
        if self.current_epoch == 0: self.plot_computation_graph()

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        """
        Validation step in main training loop (requried) TODO:
        """
        pass

    def validation_end(self, outputs):
        '''
        TODO:
        '''
        # result = dict(functools.reduce(operator.add, map(collections.Counter, outputs)))
        # for k, v in result.items(): self.logger.experiment.add_scalar("val_" + k, v, self.current_epoch)
        pass

    ### required
    def configure_optimizers(self):
        """
        Config Optimizer Wrapper (requried)
        """
        return self.model.configure_optimizers(self.params)

    """
        def sample_images(self):
            # Get sample reconstruction image
            test_input, test_label = next(iter(self.sample_dataloader))
            test_input = test_input.to(self.curr_device)
            test_label = test_label.to(self.curr_device)
            recons = self.model.generate(test_input, labels = test_label)
            vutils.save_image(recons.data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"recons_{self.logger.name}_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)
    
            # vutils.save_image(test_input.data,
            #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
            #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
            #                   normalize=True,
            #                   nrow=12)
    
            try:
                samples = self.model.sample(144,
                                            self.curr_device,
                                            labels = test_label)
                vutils.save_image(samples.cpu().data,
                                  f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                                  f"{self.logger.name}_{self.current_epoch}.png",
                                  normalize=True,
                                  nrow=12)
            except:
                pass
    
    
            del test_input, recons #, samples
    """

    def load_weights(self, ckpt_path):
        """
        load pretrained weights
        :param ckpt_path:
        :return:
        """
        if not ckpt_path: return  # no weight path
        try:
            self.load_state_dict(torch.load(ckpt_path)['state_dict'])
            print(f"{ckpt_path} Loaded!!!!!")
        except:
            # raise Exception()
            warnings.warn("CANNOT load weights")


###################################################
##### Train vanilla VAE with Human Guidance #######
###################################################


def train_feedback(epoch, model, optimizer, train_loader, train_on_gpu=True):
    """
    Train a VAE model with standard EBLO function for one single epoch
    An additional term is present is the objective, to force some points to a defined
    anchors with a MSE loss.
    The idendity of anchored points and the anchors are specified in a CSV file
    that is used by a special DataLoader.
    Please refer to human_guidance/feedback_helpers.py and class 'DSpritesDataset' for more info

    Params :
        epoch (int) : the considered epoch
        model (nn.Module) : VAE model to train
        optimizer (optim.Optimizer) : Optimizer used for VAE training
        train_loader (DataLoader) : Dataloader used for training. Need to be a custom dataloader built to
                take in account feedback from a csv file. Please refer to human_guidance/feedback_helpers.py
                and class 'DSpritesDataset' for more info

    Return the average global loss, as well as average of the different terms
    """
    # toggle model to train mode
    model.train()

    # Score the different loss per iteration (=batch)
    global_VAE_iter = []
    kl_loss_iter = []
    recon_loss_iter = []

    start = timer()

    criterion_recon = nn.BCEWithLogitsLoss().cuda()  # more stable than handmade sigmoid as last layer and BCELoss
    MSE = nn.MSELoss(reduce=False)

    def weighted_mse_loss(input, target, weight):
        return torch.sum(weight * torch.sum(MSE(input, target), dim=1))

    # each `data` is of BATCH_SIZE samples and has shape [batch_size, 4, 128, 128]
    for batch_idx, (data, _, _, feedbacks) in enumerate(train_loader):
        data = Variable(data)
        if train_on_gpu:
            data = data.cuda()

        # push whole batch of data through VAE.forward() to get recon_loss
        x_recon, mu_z, logvar_z, _ = model(data)

        # calculate scalar loss
        loss_recon = criterion_recon(x_recon, data)
        loss_recon *= data.size(1) * data.size(2) * data.size(3)
        loss_recon.div(data.size(0))

        loss_kl = kl_divergence(mu_z, logvar_z)

        # Feedbacks
        deltas = feedbacks[0].cuda().float()
        x_anchors = feedbacks[1].float()
        y_anchors = feedbacks[2].float()
        tensor_anchors = torch.cat((x_anchors, y_anchors), 1).cuda()  # BatchSize x 2
        loss_feedbacks = weighted_mse_loss(mu_z, tensor_anchors, deltas)

        loss_VAE = loss_recon + model.beta * loss_kl + loss_feedbacks

        optimizer.zero_grad()
        loss_VAE.backward()
        optimizer.step()

        global_VAE_iter.append(loss_VAE.item())
        recon_loss_iter.append(loss_recon.item())
        kl_loss_iter.append(loss_kl.item())

        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss_VAE.item()), end='\r')

    if (epoch % 10 == 0) or (epoch == 1):
        print('==========> Epoch: {} ==========> Average loss: {:.4f}'.format(epoch, np.mean(global_VAE_iter)))
        print(f'{timer() - start:.2f} seconds elapsed in epoch.')
        print(f'Reconstruction loss : {np.mean(recon_loss_iter):.2f}, KL loss : {np.mean(kl_loss_iter):.2f}')

    return np.mean(global_VAE_iter), np.mean(kl_loss_iter), np.mean(recon_loss_iter)


def train_Simple_VAE(epochs, model, optimizer, train_loader, train_on_gpu=True):
    """
    Main function to train a VAE model with standard ELBO objective function for a given number of epochs
    An additional term is present is the objective, to force some points to a defined
    anchors with a MSE loss.
    The idendity of anchored points and the anchors are specified in a CSV file
    that is used by a special DataLoader.
    Please refer to human_guidance/feedback_helpers.py and class 'DSpritesDataset' for more info
    Possible to train from scratch or resume a training (simply pass a trained VAE as input)

    The model is not saved automatically, but returned

    Params :
        epochs (int) : Maximum number of epochs
        model (nn.Module) : VAE model to train
        optimizer (optim.Optimizer) : Optimizer used for VAE training
        train_loader (DataLoader) : Dataloader used for training

    Return a pandas DataFrame containing the training history, as well as the trained model
    """

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()
    history = []

    lr_schedul_VAE = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=25, gamma=0.5)

    for epoch in range(model.epochs + 1, model.epochs + epochs + 1):
        global_VAE_loss, kl_loss, recon_loss = train_feedback(epoch, model, optimizer, train_loader, train_on_gpu=True)

        history.append([global_VAE_loss, kl_loss, recon_loss])
        model.epochs += 1

        lr_schedul_VAE.step()

    history = pd.DataFrame(
        history,
        columns=['global_VAE_loss', 'kl_loss', 'recon_loss'])

    total_time = timer() - overall_start
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    print('######### TRAINING FINISHED ##########')

    # Attach the optimizer
    model.optimizer = optimizer

    return model, history
