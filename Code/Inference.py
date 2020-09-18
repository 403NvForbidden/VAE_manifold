'''
    Profile description
'''

##########################################################
# %% imports
##########################################################
import datetime, os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline
import pickle as pkl

import torch
from torch import cuda, optim
from torchsummary import summary
from torch.autograd import Variable

from scipy.stats import norm

from models.networks import VAE, Skip_VAE, VAE2
from models.infoMAX_VAE import CNN_128_VAE, MLP_MI_estimator
from util.data_processing import get_train_val_dataloader, imshow_tensor, get_inference_dataset
from models.train_net import train_VAE_model, train_2_stage_VAE_model, train_2_stage_infoVAE_model
from util.helpers import plot_train_result, plot_train_result_info, save_checkpoint, load_checkpoint, save_brute, \
    load_brute, plot_from_csv, metadata_latent_space, save_reconstruction

##########################################################
# %% DataLoader and Co
##########################################################
datadir = '../DataSets/'
outdir = '../outputs/'

### META of dataset
datadir_BBBC = datadir + 'Synthetic_Data_1'
datadir_Horvarth = datadir + 'Peter_Horvath_Subsample'
datadir_Chaffer = datadir + 'Chaffer_Data'
dataset_path = datadir_BBBC

model_name='Model_name_string'
path_to_GT = datadir + 'MetaData1_GT_link_CP.csv'
model_path = outdir + '2stage_infoVAE_2020-09-17-23:21_100'

n = 15  # figure with 15x15 digits
digit_size = 64

# reload model
# VAE_1 = load_brute(model_path + '/VAE_1.pth')
VAE_2 = load_brute(model_path + '/VAE_2.pth')

### META of training deivice
device = torch.device('cpu' if not cuda.is_available() else 'cuda')
print(f'\tTrain on: {device}\t')

##########################################################
# %% generation and reconstruction
##########################################################
# infer_data, infer_dataloader = get_inference_dataset(dataset_path, 32, digit_size, shuffle=True, droplast=False)
# image_save = outdir + ''
# # lofi constrcution
# save_reconstruction(infer_dataloader, VAE_1, VAE_2, image_save, device, double_embed=False)

##########################################################
# %% Visualize latent space and save it
##########################################################
# display manifold of the images
figure = np.zeros((digit_size * n, digit_size * n, 3))
VAE_2.eval()
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = 1.5*norm.ppf(np.linspace(0.025, 0.925, n))
grid_y = 1.5*norm.ppf(np.linspace(0.025, 0.925, n))
grid_z = 1.5*norm.ppf(np.linspace(0.025, 0.925, n))
#grid_x = norm.ppf(np.linspace(-10.0, 10.0, n))
#grid_y = norm.ppf(np.linspace(-10.0, 10.0, n))
template = torch.randn(VAE_2.zdim)

for i, xi in enumerate(grid_x):
    for j, yi in enumerate(grid_y):
        # for k, zi in enumerate(grid_z):
        zi = grid_z[n//4]
        z_sample = torch.FloatTensor([xi, yi, zi])
        z_sample.resize_((1, VAE_2.zdim, 1, 1)) #= torch.FloatTensor(1, generator.zdim, 1, 1)
        # z_sample = np.array([np.random.uniform(-1.5, 1.5, size=generator.zdim)])
        z_sample = Variable(z_sample, requires_grad=False).to(device)
        x_decoded = VAE_2.decode(z_sample)
        x_decoded = torch.sigmoid(torch.squeeze(x_decoded)).detach().cpu()
        ### RGB channel
        x_decoded = x_decoded.permute(1, 2, 0)
        ### channel-wise : x_decoded = x_decoded[2].reshape(digit_size, digit_size)
        # x_decoded = x_decoded[1]
        figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size, :] = x_decoded

plt.figure(figsize=(25, 25))
plt.imshow(figure)
plt.savefig(model_path + '/ij_z0.25n.png')
plt.show()