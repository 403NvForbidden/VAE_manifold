# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-04-30T11:43:21+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-29T15:13:44+10:00

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
import pandas as pd
import numpy as np
import torch
import datetime
import pickle as pkl
import matplotlib.pyplot as plt
import plotly.offline

from torchsummary import summary
from torch import cuda, optim

from infoMAX_VAE import CNN_VAE, CNN_128_VAE, MLP_MI_estimator, MLP_MI_128_estimator
from data_processing import get_train_val_dataloader, imshow_tensor, get_inference_dataset
from train_net import train_InfoMAX_model, inference_recon
from helpers import plot_train_result, save_checkpoint, load_checkpoint, save_brute, load_brute, metadata_latent_space, save_reconstruction, plot_from_csv


##########################################################
# %% DataLoader and Co
##########################################################
# Location of data
datadir = 'DataSets/'
datadir_BBBC = datadir + 'Synthetic_Data_1'
datadir_Horvarth = datadir + 'Peter_Horvath_Subsample'
datadir_Chaffer = datadir + 'Chaffer_Data'
dataset_path = datadir_Chaffer ### CHANGE DATASET HERE

#path_to_GT = 'DataSets/MetaData2_PeterHorvath_GT_link_CP_SUBSAMPLE.csv'
#path_to_GT = 'DataSets/MetaData1_GT_link_CP.csv'
path_to_GT = 'DataSets/MetaData3_Chaffer_GT_link_CP.csv'

# Check if GPU avalaible
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

#Define the input size of the image
# Data will be reshape   C x input_size x input_size
input_size = 128
# Change to fit hardware
batch_size = 32
# Number of channel of the raw dataset (4 for Chaffer Dataset, 3 otherwise)
input_channel = 4

train_loader, valid_loader = get_train_val_dataloader(dataset_path,input_size,batch_size,test_split=0.15)

#Qualitative inspection of one data example
trainiter = iter(train_loader)
features, labels = next(trainiter)
_,_ = imshow_tensor(features[0])


##########################################################
# %% Build custom VAE Model and TRAIN IT
##########################################################

#### Architecture to use if input size is 64x64 ####
#VAE = CNN_VAE(zdim=3,input_channels=input_channel, alpha=20, beta=1, base_enc=64, base_dec=64)
#MLP = MLP_MI_estimator(input_size*input_size*input_channel,zdim=3)

#### Architecture to use if input size is 128x128 ####
VAE = CNN_128_VAE(zdim=3,input_channels=input_channel, alpha=20, beta=1, base_enc=64, base_dec=64)
MLP = MLP_MI_128_estimator(input_size*input_size*input_channel,zdim=3)

opti_VAE = optim.Adam(VAE.parameters(), lr=0.0001, betas=(0.9, 0.999))
opti_MLP = optim.Adam(MLP.parameters(), lr=0.0005, betas=(0.9, 0.999))

if train_on_gpu:
    VAE.cuda()
    MLP.cuda()

#summary(VAE,input_size=(input_channel,input_size,input_size),batch_size=32)

#Max number of epochs (can also early stopped if val loss not improve for a long period)
epochs = 100

### Best Model over epochs is automatically saved
model_name = f'Chaffer_InfoMax_BestModel222_{datetime.date.today()}'
save_model_path = 'temporary_save/'+f'{model_name}.pth'

VAE, MLP, history, best_epoch = train_InfoMAX_model(epochs, VAE, MLP, opti_VAE, opti_MLP, train_loader, valid_loader,saving_path=save_model_path, train_on_gpu=train_on_gpu)
fig = plot_train_result(history, best_epoch,save_path=None, infoMAX = True)
fig.show()
plt.show()

#Save training history
history_save = 'temporary_save/'+f'loss_evo_{model_name}.csv'
history.to_csv(history_save)


##########################################################
# %% Visualize latent space and save latent code in csv file
##########################################################

#Visualize on the WHOLE dataset (train & validation)
infer_data, infer_dataloader = get_inference_dataset(dataset_path,batch_size,input_size,droplast=False)

#Possibility of reloading a model trained in the past, or use the variable defined above
#model_VAE = load_brute('path_to_model.pth')
#model_name='Model_name_string'

#Where to save csv with metadata
csv_save_output = 'temporary_save/'+f'{model_name}_metedata.csv'
save_csv = True
#Store raw image data in csv (results in heavy file)
store_raw = False

metadata_csv = metadata_latent_space(VAE, infer_dataloader, train_on_gpu, GT_csv_path=path_to_GT, save_csv=save_csv, with_rawdata=store_raw,csv_path=csv_save_output)
figplotly = plot_from_csv(metadata_csv,dim=3,column='Sub_population',as_str=True)
html_save = 'temporary_save/'+f'{model_name}_Representation.html'
plotly.offline.plot(figplotly, filename=html_save, auto_open=True)

#save image of reconstruction and generated samples
image_save = 'temporary_save/'+f'{model_name}.png'
save_reconstruction(infer_dataloader,model_VAE,image_save,train_on_gpu)
