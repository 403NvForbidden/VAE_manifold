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
import torchvision
from torch import cuda, optim
from torchsummary import summary

from models.networks import VAE, Skip_VAE, VAE2, VaDE
from models.infoMAX_VAE import CNN_128_VAE
from util.Process_Mnist import get_MNIST_dataloader
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
datadir = '/mnt/Linux_Storage/'
outdir = '../outputs/'
save = False

### META of dataset
device = torch.device('cpu' if not cuda.is_available() else 'cuda')
print(f'\tTrain on: {device}\t')

### META of training
input_size = 64  # the input size of the image
batch_size = 256  # Change to fit hardware

EPOCHS = 2
train_loader, valid_loader = get_MNIST_dataloader(datadir, batch_size)
model_name = f'Vanilla_VAE_MNIST_3d'  # {datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")}'
save_model_path = None
if save:
    save_model_path = outdir + f'{model_name}/' if save else ''
    # if the dir dsnt exist
    if not os.path.isdir(save_model_path):
        os.mkdir(save_model_path)

'''
trainiter = iter(train_loader)
features, labels = next(trainiter)
_,_ = imshow_tensor(features[0])
'''

##########################################################
# %% Build custom VAE Model
##########################################################
model = VAE(zdim=3, input_channels=1).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))

model, history = train_VAE_model(EPOCHS, model, optimizer, train_loader, valid_loader, save_model_path, device)

fig = plot_train_result_GMM(history, save_path=save_model_path)
fig.show()
plt.show()