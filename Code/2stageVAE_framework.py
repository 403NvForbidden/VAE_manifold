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

from models.networks import VAE, Skip_VAE, VAE2
from models.infoMAX_VAE import CNN_128_VAE
from util.data_processing import get_train_val_dataloader, imshow_tensor, get_inference_dataset
from models.train_net import train_VAE_model, train_2stage_VAE_model
from util.helpers import plot_train_result, save_checkpoint, load_checkpoint, save_brute, load_brute, plot_from_csv, \
    metadata_latent_space, save_reconstruction

##########################################################
# %% META
##########################################################
# datadir = '/content/drive/My Drive/Colab Notebooks/Thesis/Datasets/' # colab
datadir = '../DataSets/'
outdir = '../outputs/'
save = False

### META of dataset
datadir_BBBC = datadir + 'Synthetic_Data_1'
datadir_Horvarth = datadir + 'Peter_Horvath_Subsample'
datadir_Chaffer = datadir + 'Chaffer_Data'
dataset_path = datadir_BBBC

# path_to_GT = '../DataSets/MetaData2_PeterHorvath_GT_link_CP_SUBSAMPLE.csv'
path_to_GT = datadir + 'MetaData1_GT_link_CP.csv'
# path_to_GT = '../DataSets/MetaData3_Chaffer_GT_link_CP.csv'

### META of training device
device = torch.device('cpu' if not cuda.is_available() else 'cuda')
print(f'\tTrain on: {device}\t')

### META of training
input_size = 64  # the input size of the image
batch_size = 32  # Change to fit hardware

EPOCHS = 2
train_loader, valid_loader = get_train_val_dataloader(dataset_path, input_size, batch_size, test_split=0.1)
model_name = f'2stage_VAE_{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")}'
save_model_path = None
if save:
    save_model_path = outdir + f'{model_name}_{EPOCHS}/' if save else ''
    # if the dir dsnt exist
    if not os.path.isdir(save_model_path):
        os.mkdir(save_model_path)
"""
#Qualitative inspection of one data example
trainiter = iter(train_loader)
features, labels = next(trainiter)
_,_ = imshow_tensor(features[0])
"""

##########################################################
# %% Build custom VAE Model
##########################################################
VAE_1 = VAE(zdim=100, beta=1, base_enc=32, input_channels=3, base_dec=32).to(device)
VAE_2 = VAE2(VAE_1.conv_enc, VAE_1.linear_enc, input_channels=3, zdim=3).to(device)

optimizer1 = optim.Adam(VAE_1.parameters(), lr=0.0001, betas=(0.9, 0.999))
optimizer2 = optim.Adam(VAE_2.parameters(), lr=0.0001, betas=(0.9, 0.999))

VAE_1, VAE_2, history, best_epoch = train_2stage_VAE_model(EPOCHS, VAE_1, VAE_2, optimizer1, optimizer2, train_loader,
                                                           valid_loader, save_path=save_model_path, device=device)

##########################################################
# %% Plot results
##########################################################
fig = plot_train_result(history, best_epoch, save_path=save_model_path)
fig.show()
plt.show()

# SAVE TRAINED MODEL and history
if save:
    history.to_csv(save_model_path + 'epochs.csv')

##########################################################
# %% Visualize latent space and save it
##########################################################
# Visualize on the WHOLE dataset (train & validation)
infer_data, infer_dataloader = get_inference_dataset(dataset_path, batch_size, input_size, shuffle=True, droplast=False)
"""
#Possibility of reloading a model trained in the past, or use the variable defined above
#model_VAE = load_brute(save_model_path)
#model_VAE = load_brute('path_to_model.pth')
#model_name='Model_name_string'

#Where to save csv with metadata
csv_save_output = f'{model_name}_metedata.csv'
save_csv = True
#Store raw image data in csv (results in heavy file
store_raw = False

metadata_csv = metadata_latent_space(VAE, infer_dataloader, device, GT_csv_path=path_to_GT, save_csv=save_csv, with_rawdata=store_raw,csv_path=csv_save_output)
figplotly = plot_from_csv(metadata_csv,dim=3,num_class=7)#column='Sub_population',as_str=True)
#For Chaffer Dataset
#figplotly = plot_from_csv(metadata_csv,dim=3,column='Sub_population',as_str=True)

html_save = f'{model_name}_Representation.html'
plotly.offline.plot(figplotly, filename=html_save, auto_open=True)
"""
# save image of reconstruction and generated samples
image_save = save_model_path + 'hifi'
save_reconstruction(infer_dataloader, VAE_1, VAE_2, image_save, device)
