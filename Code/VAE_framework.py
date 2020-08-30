# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-03T09:27:00+11:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-29T15:13:40+10:00

'''
Vanilla VAE and SCVAE
Main File to sequentially :
- Load and preprocess the dataset of interest
- Train a Vanilla VAE or SC VAE model with a given set of hyperparameters
- Save and plot the learnt latent representation
'''

##########################################################
# %% imports
##########################################################
import datetime
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline
import pickle as pkl

import torch
from torch import cuda, optim
from torchsummary import summary

from networks import VAE, Skip_VAE
from infoMAX_VAE import CNN_128_VAE
from data_processing import get_train_val_dataloader, imshow_tensor, get_inference_dataset
from train_net import train_VAE_model
from helpers import plot_train_result, save_checkpoint, load_checkpoint, save_brute, load_brute, plot_from_csv, metadata_latent_space, save_reconstruction


##########################################################
# %% DataLoader and Co
##########################################################
# Location of data
datadir = 'DataSets/'
datadir_BBBC = datadir + 'Synthetic_Data_1'
datadir_Horvarth = datadir + 'Peter_Horvath_Subsample'
datadir_Chaffer = datadir + 'Chaffer_Data'
dataset_path = datadir_BBBC

#path_to_GT = 'DataSets/MetaData2_PeterHorvath_GT_link_CP_SUBSAMPLE.csv'
path_to_GT = 'DataSets/MetaData1_GT_link_CP.csv'
#path_to_GT = 'DataSets/MetaData3_Chaffer_GT_link_CP.csv'

# Check if GPU avalaible
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

#Define the input size of the image
# Data will be reshape   C x input_size x input_size
input_size = 64
# Change to fit hardware
batch_size = 64

train_loader, valid_loader = get_train_val_dataloader(dataset_path,input_size,batch_size,test_split=0.15)

#Qualitative inspection of one data example
trainiter = iter(train_loader)
features, labels = next(trainiter)
_,_ = imshow_tensor(features[0])

##########################################################
# %% Build custom VAE Model
##########################################################

#model = VAE(zdim=3, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2)
model = Skip_VAE(zdim=3, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2)
if train_on_gpu:
    model.cuda()

#summary(model,input_size=(3,input_size,input_size),batch_size=32)

optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

epochs = 30

model_name = f'Chaffer_SkipVAE_b1_{datetime.date.today()}'
save_model_path = 'temporary_save/Simple_VAES/'+f'{model_name}.pth'

VAE, history, best_epoch = train_VAE_model(epochs, model, optimizer, train_loader, valid_loader,saving_path=save_model_path, train_on_gpu=train_on_gpu)
fig = plot_train_result(history, best_epoch,save_path=None, infoMAX = False)
fig.show()
plt.show()

#SAVE TRAINED MODEL and history
history_save = 'temporary_save/Simple_VAES/'+f'loss_evo_{model_name}.csv'
history.to_csv(history_save)


##########################################################
 # %% Visualize latent space and save it
##########################################################

#Visualize on the WHOLE dataset (train & validation)
infer_data, infer_dataloader = get_inference_dataset(dataset_path,batch_size,input_size,droplast=False)

#Possibility of reloading a model trained in the past, or use the variable defined above
#model_VAE = load_brute(save_model_path)
#model_VAE = load_brute('path_to_model.pth')
#model_name='Model_name_string'

#Where to save csv with metadata
csv_save_output = 'temporary_save/Simple_VAES/'+f'{model_name}_metedata.csv'
save_csv = True
#Store raw image data in csv (results in heavy file
store_raw = False

metadata_csv = metadata_latent_space(VAE, infer_dataloader, train_on_gpu, GT_csv_path=path_to_GT, save_csv=save_csv, with_rawdata=store_raw,csv_path=csv_save_output)
figplotly = plot_from_csv(metadata_csv,dim=3,num_class=7)#column='Sub_population',as_str=True)
html_save = 'temporary_save/Simple_VAES/'+f'{model_name}_Representation.html'
plotly.offline.plot(figplotly, filename=html_save, auto_open=True)

#save image of reconstruction and generated samples
image_save = 'temporary_save/Simple_VAES/'+f'{model_name}.png'
save_reconstruction(infer_dataloader,VAE,image_save,train_on_gpu)
