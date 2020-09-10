# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-04-30T11:43:21+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-31T10:48:33+10:00

'''
INFO MAX VAE
Main File to sequentially :
- Load and preprocess the dataset of interest
- Train a InfoMAX VAE model with a given set of hyperparameters
- Save and plot the learnt latent representation
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

from models.networks import VAE, Skip_VAE, VAE2
from models.infoMAX_VAE import CNN_128_VAE, MLP_MI_estimator
from util.data_processing import get_train_val_dataloader, imshow_tensor, get_inference_dataset
from models.train_net import train_VAE_model, train_2_stage_VAE_model, train_2_stage_infoVAE_model
from util.helpers import plot_train_result, save_checkpoint, load_checkpoint, save_brute, load_brute, plot_from_csv, metadata_latent_space, save_reconstruction
##########################################################
# %% DataLoader and Co
##########################################################
datadir = '../DataSets/'
outdir = '../outputs/'
save = True

### META of dataset
datadir_BBBC = datadir + 'Synthetic_Data_1'
datadir_Horvarth = datadir + 'Peter_Horvath_Subsample'
datadir_Chaffer = datadir + 'Chaffer_Data'
dataset_path = datadir_BBBC

#path_to_GT = '../DataSets/MetaData2_PeterHorvath_GT_link_CP_SUBSAMPLE.csv'
path_to_GT = datadir + 'MetaData1_GT_link_CP.csv'
#path_to_GT = '../DataSets/MetaData3_Chaffer_GT_link_CP.csv'

### META of training deivice
device = torch.device('cpu' if not cuda.is_available() else 'cuda')
print(f'\tTrain on: {device}\t')

### META of training
input_size = 64 # the input size of the image
batch_size = 32 # Change to fit hardware
input_channel = 3

EPOCHS = 1
train_loader, valid_loader = get_train_val_dataloader(dataset_path, input_size, batch_size, test_split=0.1)
model_name = f'2stage_infoVAE_{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")}'
save_model_path = outdir + f'{model_name}_{EPOCHS}/' if save else ''
# if the dir dsnt exist
if save and not os.path.isdir(save_model_path):
    os.mkdir(save_model_path)
'''
#Qualitative inspection of one data example
trainiter = iter(train_loader)
features, labels = next(trainiter)
_,_ = imshow_tensor(features[0])
'''


##########################################################
# %% Build custom VAE Model and TRAIN IT
##########################################################

#### Architecture to use if input size is 64x64 ####
VAE_1 = VAE(zdim=100, beta=1, base_enc=32, input_channels=input_channel, base_dec=32).to(device)
VAE_2 = VAE2(VAE_1.conv_enc, VAE_1.linear_enc, input_channels=input_channel, zdim=3).to(device)

MLP_1 = MLP_MI_estimator(input_size*input_size*input_channel, zdim=100).to(device)
MLP_2 = MLP_MI_estimator(input_size*input_size*input_channel, zdim=3).to(device)

#### Architecture to use if input size is 128x128 ####
#VAE = CNN_128_VAE(zdim=3,input_channels=input_channel, alpha=20, beta=1, base_enc=64, base_dec=64)
#MLP = MLP_MI_128_estimator(input_size*input_size*input_channel,zdim=3)

optimizer1 = optim.Adam(VAE_1.parameters(), lr=0.0001, betas=(0.9, 0.999))
optimizer2 = optim.Adam(VAE_2.parameters(), lr=0.0001, betas=(0.9, 0.999))
opti_MLP1 = optim.Adam(MLP_1.parameters(), lr=0.0005, betas=(0.9, 0.999))
opti_MLP2 = optim.Adam(MLP_2.parameters(), lr=0.0005, betas=(0.9, 0.999))

VAE_1, VAE_2, MLP_1, MLP_2, history, best_epoch = train_2_stage_infoVAE_model(EPOCHS, VAE_1, VAE_2, optimizer1, optimizer2, MLP_1, MLP_2, opti_MLP1, opti_MLP2, \
                                                                train_loader, valid_loader, save_path=save_model_path, device=device)
