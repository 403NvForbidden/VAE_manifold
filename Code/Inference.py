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
generator = load_brute(model_path + '/VAE_2.pth')

### META of training deivice
device = torch.device('cpu' if not cuda.is_available() else 'cuda')
print(f'\tTrain on: {device}\t')

##########################################################
# %% Visualize latent space and save it
##########################################################
# display manifold of the images
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
generator.eval()
for i in range(n):
    for j in range(n):
        z_sample = torch.FloatTensor(1, generator.zdim, 1, 1).uniform_(-3, 3)
        # z_sample = np.array([np.random.uniform(-1.5, 1.5, size=generator.zdim)])
        z_sample = Variable(z_sample, requires_grad=False).to(device)
        x_decoded = generator.decode(z_sample)
        x_decoded = torch.sigmoid(torch.squeeze(x_decoded)).detach().cpu()
        x_decoded = x_decoded[1].reshape(digit_size, digit_size)

        figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = x_decoded

plt.figure(figsize=(25, 25))
plt.imshow(figure, cmap='Greys_r')
plt.show()
plt.savefig(model_path + '/c=1.png')