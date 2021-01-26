# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-03T09:27:00+11:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-31T10:48:28+10:00

"""
conditionalVAE
Main File to sequentially :
- Load and preprocess the dataset of interest
- Train a Vanilla VAE or SC VAE model with a given set of hyperparameters
- Save and plot the learnt latent representation
"""

##########################################################
# %% imports
##########################################################
import os
from datetime import datetime

import matplotlib.pyplot as plt
import plotly.offline
import torch
from torch import cuda, optim

from models.networks import VAE, VAE2
from models.train_net import train_2stage_VAE_model
from util.data_processing import get_train_val_dataloader, get_inference_dataset
from util.helpers import plot_train_result, plot_from_csv, metadata_latent_space, load_brute, save_reconstruction, \
    conditional_gen

##########################################################
# %% META
##########################################################
# datadir = '/content/drive/My Drive/Colab Notebooks/Thesis/Datasets/' # colab
datadir = '../DataSets/'
outdir = '../outputs/'
SAVE = False
CONDITIONAL = True

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
batch_size = 64  # Change to fit hardware
"""
EPOCHS = 30
train_loader, valid_loader = get_train_val_dataloader(dataset_path, input_size, batch_size, test_split=0.15)
model_name = f'2stage_cVAE_{datetime.now().strftime("%Y-%m-%d-%H:%M")}'
save_model_path = None
if SAVE:
    save_model_path = outdir + f'{model_name}_{EPOCHS}/' if SAVE else ''
    # if the dir dsnt exist
    if not os.path.isdir(save_model_path):
        os.mkdir(save_model_path)


#Qualitative inspection of one data example
trainiter = iter(train_loader)
features, labels = next(trainiter)
_,_ = imshow_tensor(features[0])
"""

##########################################################
# %% Build custom VAE Model
##########################################################
# VAE_1 = VAE(zdim=100, alpha=1, beta=1, input_channels=4, conditional=CONDITIONAL).to(device)
# VAE_2 = VAE2(VAE_1.conv_enc, VAE_1.linear_enc, input_channels=4, zdim=3, conditional=CONDITIONAL).to(device)
#
# optimizer1 = optim.Adam(VAE_1.parameters(), lr=0.0001, betas=(0.9, 0.999))
# optimizer2 = optim.Adam(VAE_2.parameters(), lr=0.0001, betas=(0.9, 0.999))
#
# VAE_1, VAE_2, history, best_epoch = train_2stage_VAE_model(EPOCHS, VAE_1, VAE_2, optimizer1, optimizer2, train_loader,
#                                                            valid_loader, save_path=save_model_path, device=device)
#
# ##########################################################
# # %% Plot results
# ##########################################################
# fig = plot_train_result(history, best_epoch, save_path=save_model_path)
# fig.show()
# plt.show()
#
# # SAVE TRAINED MODEL and history
# if SAVE:
#     history.to_csv(save_model_path + 'epochs.csv')

##########################################################
# %% Visualize latent space and save it
##########################################################
# Visualize on the WHOLE dataset (train & validation)

infer_data, infer_dataloader = get_inference_dataset(dataset_path, batch_size, input_size, shuffle=True, droplast=False)

save_model_path = "../outputs/2stage_cVAE_2020-10-08-20:38_30/"
VAE_1 = load_brute(save_model_path + 'VAE_1.pth')
VAE_2 = load_brute(save_model_path + 'VAE_2.pth')
VAE_2.conditioanl = True

conditional_gen(infer_dataloader, VAE_1, VAE_2, save_model_path, device)
"""
# Where to save csv with metadata
csv_save_output = f'{save_model_path}/metedata.csv'
save_csv = True
# Store raw image data in csv (results in heavy file
store_raw = False

metadata_csv = metadata_latent_space(VAE_2, infer_dataloader, device, GT_csv_path=path_to_GT, save_csv=save_csv,
                                     with_rawdata=store_raw, csv_path=csv_save_output)
figplotly = plot_from_csv(metadata_csv, dim=3, num_class=7)  # column='Sub_population',as_str=True)
# For Chaffer Dataset
# figplotly = plot_from_csv(metadata_csv,dim=3,column='Sub_population',as_str=True)

html_save = f'{save_model_path}/Representation.html'
plotly.offline.plot(figplotly, filename=html_save, auto_open=True)

# save image of reconstruction and generated samples

save_reconstruction(infer_dataloader, VAE_1, VAE_2, save_model_path, device, gen=True)
"""