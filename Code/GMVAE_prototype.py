# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-03T09:27:00+11:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-31T10:48:28+10:00

"""
Vanilla VAE and SCVAE
Main File to sequentially :
- Load and preprocess the dataset of interest
- Train a Vanilla VAE or SC VAE model with a given set of hyperparameters
- Save and plot the learnt latent representation
"""

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

from models.networks import VAE, Skip_VAE, VAE2, VaDE
from models.infoMAX_VAE import CNN_128_VAE
from util.data_processing import get_train_val_dataloader, imshow_tensor, get_inference_dataset
from models.train_net import train_VAE_model, train_2stage_VAE_model, train_vaDE_model
from util.helpers import plot_train_result, save_checkpoint, load_checkpoint, save_brute, load_brute, plot_from_csv, \
    metadata_latent_space, save_reconstruction, plot_train_result_GMM

##########################################################
# %% META
##########################################################
# datadir = '/content/drive/My Drive/Colab Notebooks/Thesis/Datasets/' # colab
datadir = '../DataSets/'
outdir = '../outputs/'
save = True

### META of dataset
datadir_BBBC = datadir + 'Synthetic_Data_1'
datadir_Horvarth = datadir + 'Peter_Horvath_Subsample'
datadir_Chaffer = datadir + 'Chaffer_Data'
dataset_path = datadir_Chaffer

# path_to_GT = '../DataSets/MetaData2_PeterHorvath_GT_link_CP_SUBSAMPLE.csv'
# path_to_GT = datadir + 'MetaData1_GT_link_CP.csv'
path_to_GT = '../DataSets/MetaData3_Chaffer_GT_link_CP.csv'

### META of training device
device = torch.device('cpu' if not cuda.is_available() else 'cuda')
print(f'\tTrain on: {device}\t')

### META of training
input_size = 128  # the input size of the image
batch_size = 64  # Change to fit hardware

EPOCHS = 30
train_loader, valid_loader = get_train_val_dataloader(dataset_path, input_size, batch_size, test_split=0.1)
model_name = f'VaDE_test_Chaffer_Data'  # {datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")}'
save_model_path = None
if save:
    save_model_path = outdir + f'{model_name}/' if save else ''
    # if the dir dsnt exist
    if not os.path.isdir(save_model_path):
        os.mkdir(save_model_path)
"""
### Qualitative inspection of one data example
trainiter = iter(train_loader)
features, labels = next(trainiter)
_,_ = imshow_tensor(features[0])
"""

##########################################################
# %% Build custom VAE Model
##########################################################
model = VaDE(zdim=3, ydim=12, input_channels=4).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))

model, history = train_vaDE_model(EPOCHS, model, optimizer, train_loader, valid_loader, save_model_path, device)

##########################################################
# %% Visualize latent space and save it
##########################################################
fig = plot_train_result_GMM(history, save_path=save_model_path)
fig.show()
plt.show()

# SAVE TRAINED MODEL and history
if save:
    history.to_csv(save_model_path + 'epochs.csv')

##########################################################
# %% Visualize latent space and save it
##########################################################