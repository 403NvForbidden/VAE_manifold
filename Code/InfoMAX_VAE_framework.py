# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-04-30T11:43:21+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-31T10:48:33+10:00

"""
INFO MAX VAE
Main File to sequentially :
- Load and preprocess the dataset of interest
- Train a InfoMAX VAE model with a given set of hyperparameters
- Save and plot the learnt latent representation
"""

##########################################################
# %% imports
##########################################################
import datetime
import os

import matplotlib.pyplot as plt
import torch
from torch import cuda, optim

from models.infoMAX_VAE import MLP_MI_estimator
from models.networks import VAE, VAE2
from models.train_net import train_2stage_infoMaxVAE_model
from util.data_processing import get_train_val_dataloader, get_inference_dataset
from util.helpers import plot_train_result_infoMax, save_reconstruction

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

# path_to_GT = '../DataSets/MetaData2_PeterHorvath_GT_link_CP_SUBSAMPLE.csv'
path_to_GT = datadir + 'MetaData1_GT_link_CP.csv'
# path_to_GT = '../DataSets/MetaData3_Chaffer_GT_link_CP.csv'

### META of training deivice
device = torch.device('cpu' if not cuda.is_available() else 'cuda')
print(f'\tTrain on: {device}\t')

### META of training
input_size = 64  # the input size of the image
batch_size = 32  # Change to fit hardware
input_channel = 3
double_embed = False  # the variable that allows MI(z1, z2)
input_compress = 100 if double_embed else 3
EPOCHS = 50

train_loader, valid_loader = get_train_val_dataloader(dataset_path, input_size, batch_size, test_split=0.1)
model_name = f'2stage_infoMaxVAE_{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")}'
if double_embed:
    model_name = f'2stage_double_embed_infoMaxVAE_{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")}'

save_model_path = outdir + f'{model_name}_{EPOCHS}/' if save else ''

# if the dir dsnt exist
if save and not os.path.isdir(save_model_path):
    os.mkdir(save_model_path)
##########################################################
# %% Build custom VAE Model and TRAIN IT
##########################################################

### Architecture to use if input size is 64x64
VAE_1 = VAE(zdim=100, alpha=500, beta=100, input_channels=input_channel).to(device)
VAE_2 = VAE2(VAE_1.conv_enc, VAE_1.linear_enc, zdim=3, alpha=500, beta=100, input_channels=input_compress,
             double_embed=double_embed).to(device)

MLP_1 = MLP_MI_estimator(input_dim=input_size * input_size * input_channel, zdim=100).to(device)
if double_embed:
    MLP_2 = MLP_MI_estimator(input_dim=100, zdim=3).to(device)
else:
    MLP_2 = MLP_MI_estimator(input_dim=input_size * input_size * input_channel, zdim=3).to(device)
### Architecture to use if input size is 128x128
# VAE = CNN_128_VAE(zdim=3,input_channels=input_channel, alpha=20, beta=1, base_enc=64, base_dec=64)
# MLP = MLP_MI_128_estimator(input_size*input_size*input_channel,zdim=3)

optimizer1 = optim.Adam(VAE_1.parameters(), lr=0.0001, betas=(0.9, 0.999))
optimizer2 = optim.Adam(VAE_2.parameters(), lr=0.0001, betas=(0.9, 0.999))
opti_MLP1 = optim.Adam(MLP_1.parameters(), lr=0.0005, betas=(0.9, 0.999))
opti_MLP2 = optim.Adam(MLP_2.parameters(), lr=0.0005, betas=(0.9, 0.999))

VAE_1, VAE_2, MLP_1, MLP_2, history, best_epoch = train_2stage_infoMaxVAE_model(EPOCHS, VAE_1, VAE_2, optimizer1,
                                                                                optimizer2, MLP_1, MLP_2, opti_MLP1,
                                                                                opti_MLP2, train_loader, valid_loader,
                                                                                gamma=0.8,
                                                                                save_path=save_model_path,
                                                                                double_embed=double_embed,
                                                                                device=device)
##########################################################
# %% Plot results
##########################################################
fig = plot_train_result_infoMax(history, best_epoch, save_path=save_model_path)
fig.show()
plt.show()

# SAVE TRAINED MODEL and history
if save:
    history.to_csv(save_model_path + 'epochs.csv')

##########################################################
# %% Visualize latent space and save it
##########################################################
# Visualize on the WHOLE dataset (train & validation)
infer_data, infer_dataloader = get_inference_dataset(dataset_path, batch_size, input_size, shuffle=False,
                                                     droplast=False)
"""
#Possibility of rel oading a model trained in the past, or use the variable defined above
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
image_save = save_model_path + ''
# lofi constrcution
save_reconstruction(infer_dataloader, VAE_1, VAE_2, image_save, device, double_embed=double_embed)