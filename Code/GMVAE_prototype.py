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

"""
https://github.com/mori97/VaDE
https://github.com/GuHongyang/VaDE-pytorch
https://github.com/jariasf/GMVAE
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
    metadata_latent_space, save_reconstruction, plot_train_result_GMM, metadata_latent_space_single
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
from PIL import Image
##########################################################
# %% META
##########################################################
# datadir = '/content/drive/My Drive/Colab Notebooks/Thesis/Datasets/' # colab
datadir = '../DataSets/'
outdir = '../outputs/'
save = True

### META of dataset
datadir_BBBC = datadir + 'Synthetic_Data_1'
datadir_Horvarth = datadir + 'Selected_Hovarth' #'Peter_Horvath_Subsample'
datadir_Chaffer = datadir + 'Selected_Chaffer'
dataset_path = datadir_Horvarth

path_to_GT = '../DataSets/MetaData2_PeterHorvath_GT_link_CP_SUBSAMPLE.csv'
# path_to_GT = datadir + 'MetaData1_GT_link_CP.csv'
# path_to_GT = '../DataSets/MetaData3_Chaffer_GT_link_CP.csv'

### META of training device
device = torch.device('cpu' if not cuda.is_available() else 'cuda')
print(f'\tTrain on: {device}\t')

### META of training
input_size = 64  # the input size of the image
batch_size = 128  # Change to fit hardware

EPOCHS = 80
train_loader, valid_loader = get_train_val_dataloader(dataset_path, input_size, batch_size, test_split=0.1)
model_name = f'VaDE_Selected_Hovarth_3d'  # {datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")}'
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
model = VaDE(zdim=3, ydim=3, input_channels=3).to(device)

#
# optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
#
# model, history = train_vaDE_model(EPOCHS, model, optimizer, train_loader, valid_loader, save_model_path, device)
#
# ##########################################################
# # %% Visualize latent space and save it
# ##########################################################
# fig = plot_train_result_GMM(history, save_path=save_model_path)
# fig.show()
# plt.show()
#
# # SAVE TRAINED MODEL and history
# if save:
#     history.to_csv(save_model_path + 'epochs.csv')


##########################################################
# %% Visualize latent space and save it

# load model
model.load_state_dict(torch.load(save_model_path + 'model.pth'))

sample = Variable(torch.randn(15, 3, 1, 1), requires_grad=False).to(device)
recon = model.decoder(sample)

img_grid = make_grid(torch.sigmoid(recon[:, :3, :, :]),
                             nrow=4, padding=12, pad_value=1)
plt.figure(figsize=(25, 16))
plt.imshow(img_grid.detach().cpu().permute(1, 2, 0))
plt.axis('off')
plt.title(f'Random generated samples')
plt.show()
# plt.savefig(pre'generatedSamples.png')

# %%
# Visualize on the WHOLE dataset (train & validation)
infer_data, infer_dataloader = get_inference_dataset(dataset_path, batch_size, input_size, shuffle=False,
                                                     droplast=False)
metadata_csv = metadata_latent_space_single(model, infer_dataloader=infer_dataloader, device=device, GT_csv_path=path_to_GT, save_csv=True, with_rawdata=False,csv_path=save_model_path + 'metadata_csv')
''''''

# %%
# load existing local csv fiels
metadata_csv = pd.read_csv(save_model_path + 'metadata_csv.csv').dropna().reindex()
# %%
figplotly = plot_from_csv(metadata_csv, dim=3, num_class=3) # column='Sub_population',as_str=True)
#For Chaffer Dataset
#figplotly = plot_from_csv(metadata_csv,dim=3,column='Sub_population',as_str=True)

html_save = f'{save_model_path}_Representation.html'
plotly.offline.plot(figplotly, filename=html_save, auto_open=True)

# %% Sample from a cluster
cluster = 2
# zzz = torch.randn(3, 100, 1, 1)
sample = Variable(model.mu_c.permute(1, 0).unsqueeze(-1).unsqueeze(-1), requires_grad=False).to(device)

recon = model.decoder(sample)

img_grid = make_grid(torch.sigmoid(recon[:, :3, :, :]),
                             nrow=4, padding=12, pad_value=1)
plt.figure(figsize=(25, 16))
plt.imshow(img_grid.detach().cpu().permute(1, 2, 0))
plt.axis('off')
plt.title(f'Random generated samples')
plt.show()