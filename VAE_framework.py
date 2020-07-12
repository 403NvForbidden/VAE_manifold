# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-03T09:27:00+11:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-07-11T18:35:57+10:00

##########################################################
# %% imports
##########################################################

from networks import VAE, Skip_VAE
from torchsummary import summary
from torch import cuda, optim
from data_processing import get_train_val_dataloader, image_tranforms, imshow_tensor, get_inference_dataset
from train_net import train_VAE_model, inference_recon
from helpers import plot_train_result, save_checkpoint, load_checkpoint, save_brute, load_brute, plot_from_csv, metadata_latent_space, save_reconstruction
import torch
import datetime
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.offline
import gc

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
batch_size = 2048

train_loader, valid_loader = get_train_val_dataloader(datadir_BBBC,input_size,batch_size,test_split=0.15)

#Qualitative inspection of one data example
trainiter = iter(train_loader)
features, labels = next(trainiter)
_,_ = imshow_tensor(features[0])

##########################################################
# %% Build custom VAE Model
##########################################################


input_size = 64
counter = 0
while batch_size > 0 :
    try:
        # Change to fit hardware
        batch_size = batch_size

        train_loader, valid_loader = get_train_val_dataloader(datadir_BBBC,input_size,batch_size,test_split=0.15)


        #model = VAE(zdim=3, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2)
        model = Skip_VAE(zdim=3, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2)
        if train_on_gpu:
            model.cuda()

        #print(model)
        #summary(model,input_size=(3,input_size,input_size),batch_size=32)

        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

        epochs = 30

        model_name = f'3chan_dataset1_3z_SkipVAE_{datetime.date.today()}'
        save_model_path = 'temporary_save/'+f'{model_name}.pth'

        VAE, history, best_epoch = train_VAE_model(epochs, model, optimizer, train_loader, valid_loader,saving_path=save_model_path,each_iteration=False, train_on_gpu=train_on_gpu)
        fig = plot_train_result(history, best_epoch,save_path=None, infoMAX = False)
        fig.show()
        plt.show()

        break
    except RuntimeError as e:
        if 'out of memory' in str(e):
            batch_size = int(batch_size/2)
            print(f'Batch Size is too big - reduce it to {batch_size}')
            counter +=1
            print(f'loop {counter}')
            with torch.no_grad():

                del model
                del optimizer
                gc.collect()
                torch.cuda.empty_cache()
            continue

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
model_VAE = load_brute('temporary_save/3chan_dataset1_3z_VanillaVAE_2020-07-11_VAE_.pth')
model_name='3chan_dataset1_3z_InfoMAX_2020-06-19_VAE_'
#Where to save csv with metadata
csv_save_output = 'temporary_save/'+f'{model_name}_metedata.csv'
save_csv = False
#Store raw image data in csv (results in heavy file, but raw data is needed for some metrics)
store_raw = False

metadata_csv = metadata_latent_space(model_VAE, infer_dataloader, train_on_gpu, GT_csv_path=path_to_GT, save_csv=save_csv, with_rawdata=store_raw,csv_path=csv_save_output)
figplotly = plot_from_csv(metadata_csv,dim=3,num_class=7)
html_save = 'temporary_save/'+f'{model_name}_Representation.html'
plotly.offline.plot(figplotly, filename=html_save, auto_open=True)

#save image of reconstruction and generated samples
image_save = 'temporary_save/'+f'{model_name}.png'
save_reconstruction(infer_dataloader,model_VAE,image_save,train_on_gpu)



##########################################################
# %% Load an existing model and continue to train it (or make pred)
##########################################################

date = '2020-04-17'
model_name = 'testMODEL_z2_e300'

load_model_path = 'outputs/saved_models/'+f'VAE_{model_name}_{date}.pth'

model = load_brute(load_model_path)


# %%
#SEE THE RECONSTRUCTION FOR RANDOM SAMPLE OF VAL DATASET
inference_recon(model, dataloader['val'], 16, train_on_gpu)


# %%
# PLOT THE LATENT SPACE OF THE Model
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
fig = plot_latent_space(model,dataloader['train'],train_on_gpu)
plt.show(block=True)
