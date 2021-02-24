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
import plotly.offline
from models.train_net import train_vaDE_model
from quantitative_metrics.performance_metrics_single import compute_perf_metrics
from util.config import dataset_lookUp
from util.helpers import metadata_latent_space, plot_from_csv

"""
https://github.com/mori97/VaDE
https://github.com/GuHongyang/VaDE-pytorch
https://github.com/jariasf/GMVAE
"""

##########################################################
# %% imports
##########################################################
import os
import pandas as pd

import torch
from torch import cuda, optim

from models.networks_refactoring import VaDE
from util.data_processing import get_train_val_dataloader, get_inference_dataset

##########################################################
# %% META
##########################################################
# datadir = '/content/drive/My Drive/Colab Notebooks/Thesis/Datasets/' # colab
datadir = '../DataSets/'
outdir = '/mnt/Linux_Storage/outputs/'
save = True

### META of dataset
datadir_BBBC = datadir + 'Synthetic_Data_1'
datadir_Horvarth = datadir + 'Selected_Hovarth'  # 'Peter_Horvath_Subsample'
datadir_Chaffer = datadir + 'Selected_Chaffer'
dataset_path = datadir_BBBC

# path_to_GT = '../DataSets/MetaData2_PeterHorvath_GT_link_CP_SUBSAMPLE.csv'
path_to_GT = datadir + 'MetaData1_GT_link_CP.csv'
# path_to_GT = '../DataSets/MetaData3_Chaffer_GT_link_CP.csv'

### META of training device
device = torch.device('cpu' if not cuda.is_available() else 'cuda')
print(f'\tTrain on: {device}\t')

### META of training
input_size = 64  # the input size of the image
batch_size = 128  # Change to fit hardware

EPOCHS = 35
train_loader, valid_loader = get_train_val_dataloader(dataset_path, input_size, batch_size, test_split=0.1)
model_name = f'VaDE_BBBC'  # {datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")}'
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
if os.path.exists(save_model_path + 'model.pth'):
    model.load_state_dict(torch.load(save_model_path + 'model.pth'))
else:
    optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    model, history = train_vaDE_model(EPOCHS, model, optimizer, train_loader, valid_loader, save_model_path, device)

'''
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

# load model
# model.load_state_dict(torch.load(save_model_path + 'model.pth'))

sample = Variable(torch.randn(15, 50, 1, 1), requires_grad=False).to(device)
recon = model.decoder(sample)

img_grid = make_grid(torch.sigmoid(recon[:, :3, :, :]),
                     nrow=4, padding=12, pad_value=1)
plt.figure(figsize=(25, 16))
plt.imshow(img_grid.detach().cpu().permute(1, 2, 0))
plt.axis('off')
plt.title(f'Random generated samples')
plt.show()
plt.savefig(save_model_path + 'generatedSamples.png')

''''''
# %%
# Visualize on the WHOLE dataset (train & validation)
infer_data, infer_dataloader = get_inference_dataset(dataset_path, batch_size, input_size, shuffle=False,
                                                     droplast=False)
metadata_csv = metadata_latent_space(model, infer_dataloader, device=device,
                                            GT_csv_path=path_to_GT, save_csv=True, with_rawdata=False,
                                            csv_path=save_model_path + 'metadata_csv.csv')



# %%
# load existing local csv fiels
metadata_csv = pd.read_csv(save_model_path + 'metadata_csv.csv').dropna().reindex()


# %%
figplotly = plot_from_csv(metadata_csv, ['z0', 'z1', 'z2'], dim=3, num_class=3)  # column='Sub_population',as_str=True)
# For Chaffer Dataset
# figplotly = plot_from_csv(metadata_csv,dim=3,column='Sub_population',as_str=True)

html_save = f'{save_model_path}_Representation.html'
plotly.offline.plot(figplotly, filename=html_save, auto_open=True)

# %% Sample from a cluster
cluster = 2
# zzz = torch.randn(3, 100, 1, 1)
sample = Variable(model.mu_c.permute(1, 0).unsqueeze(-1).unsqueeze(-1), requires_grad=False).to(device)

recon = model.d ecoder(sample)

img_grid = make_grid(torch.sigmoid(recon[:, :3, :, :]),
                             nrow=4, padding=12, pad_value=1)
plt.figure(figsize=(25, 16))
plt.imshow(img_grid.detach().cpu().permute(1, 2, 0))
plt.axis('off')
plt.title(f'Random generated samples')
plt.show()

'''
# %% Qualitative test
metadata_csv = pd.read_csv(save_model_path + 'metadata_csv.csv').dropna().reindex()
params_preferences = {
    'feature_size': 64 * 64 * 3,
    'path_to_raw_data':'../DataSets/Synthetic_Data_1',
    # 'path_to_raw_data': '../DataSets/Selected_Hovarth',
    'dataset_tag': 1,  # 1:BBBC 2:Horvath 3:Chaffer
    'low_dim_names': [f'z{n}' for n in range(model.zdim)], #['VAE_x_coord', 'VAE_y_coord', 'VAE_z_coord'],
    'global_saving_path': save_model_path,  # Different for each model, this one is update during optimization

    ### Unsupervised metrics
    'save_unsupervised_metric': False,
    'only_local_Q': False,
    'kt': 300,
    'ks': 500,

    ### Mutual Information
    'save_mine_metric': False,
    'batch_size': 256,
    'bound_type': 'interpolated',
    'alpha_logit': -4.6,  # result in alpha = 0.01
    'epochs': 10,

    ### Classifier accuracy
    'save_classifier_metric': True,
    'num_iteration': 3,

    ### BackBone Metric
    'save_backbone_metric': False,

    ### Disentanglement Metric
    'save_disentanglement_metric': False,
    'features': dataset_lookUp['BBBC']['feat'],
}
compute_perf_metrics(metadata_csv, params_preferences)

