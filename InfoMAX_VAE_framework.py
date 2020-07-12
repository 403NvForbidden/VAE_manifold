# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-04-30T11:43:21+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-07-13T09:06:26+10:00


##########################################################
# %% imports
##########################################################
#%matplotlib qt
from infoMAX_VAE import CNN_VAE, MLP_MI_estimator
from torchsummary import summary
from torch import cuda, optim
from data_processing import get_train_val_dataloader, image_tranforms, imshow_tensor, get_inference_dataset
from train_net import train_InfoMAX_model, inference_recon
from helpers import plot_train_result, save_checkpoint, load_checkpoint, save_brute, load_brute, plot_latent_space, metadata_latent_space, save_reconstruction
import torch
import datetime
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.offline
#plt.ion()

##########################################################
# %% DataLoader and Co
##########################################################
# Location of data
datadir = 'DataSets/'
datadir_BBBC = datadir + 'Synthetic_Data_1'
datadir_Horvarth = datadir + 'Peter_Horvath_Data'
dataset_path = datadir_BBBC

#path_to_GT = 'DataSets/MetaData2_PeterHorvath_GT_link_CP.csv'
path_to_GT = 'DataSets/MetaData1_GT_link_CP.csv'

# Check if GPU avalaible
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

#Define the input size of the image
# Data will be reshape   C x input_size x input_size
input_size = 64
# Change to fit hardware
batch_size = 256

train_loader, valid_loader = get_train_val_dataloader(datadir_BBBC,input_size,batch_size,test_split=0.15)

#Qualitative inspection of one data example
trainiter = iter(train_loader)
features, labels = next(trainiter)
_,_ = imshow_tensor(features[0])


##########################################################
# %% Build custom VAE Model and TRAIN IT
##########################################################

VAE = CNN_VAE(zdim=3, alpha=15, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2)
MLP = MLP_MI_estimator(64*63*3,zdim=3)


opti_VAE = optim.Adam(VAE.parameters(), lr=0.0001, betas=(0.9, 0.999))
opti_MLP = optim.Adam(MLP.parameters(), lr=0.0005, betas=(0.9, 0.999))

if train_on_gpu:
    VAE.cuda()
    MLP.cuda()

summary(VAE,input_size=(3,64,64),batch_size=32)

#Max number of epochs (can also early stopped if val loss not improve for a long period)
epochs = 150

model_name = f'3chan_dataset1_3z_InfoMAX_{datetime.date.today()}'
save_model_path = 'temporary_save/'+f'{model_name}.pth'

VAE, MLP, history, best_epoch = train_InfoMAX_model(epochs, VAE, MLP, opti_VAE, opti_MLP, train_loader, valid_loader,saving_path=save_model_path,each_iteration=False, train_on_gpu=train_on_gpu)
fig = plot_train_result(history, best_epoch,save_path=None, infoMAX = True)
fig.show()
plt.show()

#SAVE TRAINED MODEL and history
history_save = 'temporary_save/'+f'loss_evo_{model_name}.csv'
history.to_csv(history_save)

##########################################################
# %% Visualize latent space and save it
##########################################################

#Visualize on the WHOLE dataset (train & validation)
infer_data, infer_dataloader = get_inference_dataset(datadir_BBBC,batch_size,input_size,droplast=False)
#Load the model that has been trained above
#model_VAE = load_brute(save_model_path)
model_VAE = load_brute('temporary_save/3chan_dataset1_3z_InfoMAX_2020-06-19_VAE_.pth')
model_name='3chan_dataset1_3z_InfoMAX_2020-06-19_VAE_'
#Where to save csv with metadata
csv_save_output = 'temporary_save/'+f'{model_name}_metedata.csv'
save_csv = False
#Store raw image data in csv (results in heavy file, but raw data is needed for some metrics)
store_raw = False

metadata_csv = metadata_latent_space(model_VAE, infer_dataloader, train_on_gpu, GT_csv_path=path_to_GT, save_csv=save_csv, with_rawdata=store_raw,csv_path=csv_save_output)
figplotly = plot_from_csv(metadata_csv,dim=3,num_class=7)
html_save = 'temporary_save/'+f'{model_name}_Representation.html'
plotly.offline.plot(figplotly, filename=html_save, auto_open=False)

#save image of reconstruction and generated samples
image_save = 'temporary_save/'+f'{model_name}.png'
save_reconstruction(infer_dataloader,model_VAE,image_save,train_on_gpu)

##########################################################
# %% Visualize training history
##########################################################
#
# history_load = 'outputs/plot_history/'+f'loss_evo_{model_name}_{datetime.date.today()}.pkl'
# with open(history_load, 'rb') as f:
#     history = pkl.load(f)
#
# plot_train_result(history)
#
#
#
# ##########################################################
# # %% Load an existing model and continue to train it (or make pred)
# ##########################################################
# INFERENCE LATENT REPRESENTATION PLOT
# batch_size = 128
# input_size = 64
# dataset_name = 'Synthetic_Data_1'
# infer_data, infer_dataloader = get_inference_dataset('DataSets/'+dataset_name,batch_size,input_size,droplast=False)
#
#
#
# model_VAE = load_brute('temporary_save/3chan_dataset1_3z_InfoMAX_2020-06-19_VAE_.pth')
#
# #Path to CSV that contains GT of the dataset
# #path_to_GT = 'DataSets/MetaData2_PeterHorvath_GT_link_CP.csv'
# path_to_GT = 'DataSets/MetaData1_GT_link_CP.csv'
# #Where to save csv with metadata
# csv_save_output = 'DataSets/John_Metadata_PeterHorvarth_VAELatentCode_20200610.csv'
# save_csv = False
# #Store raw image data in csv (results in heavy file, but raw data is needed for some metrics)
# store_raw = False
# figplotly = metadata_latent_space(model_VAE, infer_dataloader, train_on_gpu, GT_csv_path=path_to_GT, save_csv=save_csv, with_rawdata=store_raw,csv_path=csv_save_output)
#
# import plotly.offline
# plotly.offline.plot(figplotly, filename='test.html', auto_open=True)
