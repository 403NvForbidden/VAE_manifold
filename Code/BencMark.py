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
from matplotlib.gridspec import GridSpec
from torch import cuda, optim
from torchsummary import summary

from models.networks import VAE, Skip_VAE, VAE2, VaDE
from models.infoMAX_VAE import CNN_128_VAE
from util.Process_Mnist import get_MNIST_dataloader
from util.data_processing import get_train_val_dataloader, imshow_tensor, get_inference_dataset
from models.train_net import train_VAE_model, train_2stage_VAE_model, train_vaDE_model
from util.helpers import plot_singleVAE_result, meta_MNIST, plot_train_result_GMM
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
from PIL import Image

##########################################################
# %% META
##########################################################

# datadir = '/content/drive/My Drive/Colab Notebooks/Thesis/Datasets/' # colab
datadir = '/mnt/Linux_Storage/'
outdir = '../outputs/'
save = True

### META of dataset
device = torch.device('cpu' if not cuda.is_available() else 'cuda')
print(f'\tTrain on: {device}\t')

### META of training
input_size = 64  # the input size of the image
batch_size = 128  # Change to fit hardware

EPOCHS = 50
train_loader, valid_loader = get_MNIST_dataloader(datadir, batch_size)
model_name = f'VaDE_MNIST_3d'  # {datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")}'
save_model_path = None
if save:
    save_model_path = outdir + f'{model_name}/' if save else ''
    # if the dir dsnt exist
    if not os.path.isdir(save_model_path):
        os.mkdir(save_model_path)


# trainiter = iter(train_loader)
# features, labels = next(trainiter)
# _,_ = imshow_tensor(features[0])
##########################################################
# %% Build custom VAE Model
##########################################################
model = VaDE(zdim=3, ydim=10, input_channels=1).to(device)
# model = VAE(zdim=3, input_channels=1).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))

# model, history = train_VAE_model(EPOCHS, model, optimizer, train_loader, valid_loader, save_model_path, device)
model, history = train_vaDE_model(EPOCHS, model, optimizer, train_loader, valid_loader, save_model_path, device)

# %%
# fig = plot_singleVAE_result(history, save_path=save_model_path)
fig = plot_train_result_GMM(history, save_path=save_model_path)
fig.show()
plt.show()

##########################################################
# %% 3D plot
##########################################################
# potentially load the model
model.load_state_dict(torch.load(save_model_path + 'model.pth'))

figplotly = meta_MNIST(model, valid_loader, device=device)

html_save = f'{save_model_path}Representation.html'
plotly.offline.plot(figplotly, filename=html_save, auto_open=True)
'''
##########################################################
# %% Qualitative Evaluation
##########################################################
cluster = 2
# zzz = torch.randn(3, 100, 1, 1)
std =model.log_sigma2_c * 0.1 + model.mu_c
sample = Variable(std.permute(1, 0).unsqueeze(-1).unsqueeze(-1), requires_grad=False).to(device)

recon = model.decoder(sample)

img_grid = make_grid(torch.sigmoid(recon[:, :3, :, :]),
                             nrow=4, padding=12, pad_value=1)
plt.figure(figsize=(25, 16))
plt.imshow(img_grid.detach().cpu().permute(1, 2, 0))
plt.axis('off')
plt.title(f'Random generated samples')
plt.show()
##########################################################
# %% Quantitative evaluation
##########################################################

'''