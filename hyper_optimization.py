# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-06-12T09:10:13+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-06-28T11:12:49+10:00

'''
File running and saving results of several training of same models with
different hyperparameters (in a Grid Search Manner)
'''

import datetime
import os
import errno
import pandas as pd
import numpy as np
import torch
from torchsummary import summary
from torch import cuda, optim
import shutil

from data_processing import get_train_val_dataloader,get_inference_dataset
from train_net import train_InfoMAX_model, inference_recon
from helpers import plot_train_result, save_brute, load_brute, plot_latent_space, metadata_latent_space, save_reconstruction
from infoMAX_VAE import CNN_VAE, MLP_MI_estimator
import plotly.offline
import matplotlib.pyplot as plt
#####################################
#### Define main variable ###########
#####################################
EXPERIMENT_TAG = f'run2_MIupgrade_alpha-beta_{datetime.date.today()}'
DOC_TEXT = []

MODEL_TAG = 3 # 1: Vanilla 2: SKIP-connection 3: InfoMax 4: InfoMAX + SkipConnection
DATASET_TAG = 1 # 1: BBBC Synthetic Dataset 2: Peter Horvarth Synthetic Dataset 3: Christine Chaffer real Dataset
DESCRIPTION_STR = 'Optimization of aplha and beta weight on objective function. Model Architecture is fixed'


#saving arborization
switcher_model = {
    1:'Vanilla_VAE',
    2:'SKIP_Co_VAE',
    3:'InfoMAX_VAE',
    4:'SKIP_and_InfoMAX_VAE'
}
switcher_dataset = {
    1:'Dataset1',
    2:'Dataset2',
    3:'Dataset3'
}
DOC_TEXT.append(('Experiment Tag :',EXPERIMENT_TAG,
                'Model :', switcher_model.get(MODEL_TAG),
                'Dataset :', switcher_dataset.get(DATASET_TAG),
                'Description of run :', DESCRIPTION_STR
))

# %%
save_folder = f'optimization/{switcher_model.get(MODEL_TAG)}/{switcher_dataset.get(DATASET_TAG)}/'
try:
    os.mkdir(save_folder+EXPERIMENT_TAG)
    save_folder = save_folder+EXPERIMENT_TAG+'/'
except FileExistsError as e:
    raise FileExistsError('!! Experiment tag folder already exists, risk of overwriting important data !!') from e


# %%
########################################
#### Prepare VAE Framework #############
########################################
switcher_datadir = {
    1:'DataSets/Synthetic_Data_1',
    2:'DataSets/Peter_Horvath_Data',
    3:'Dataset/NODIRYET'
}
datadir = switcher_datadir.get(DATASET_TAG)

switcher_path_to_GT = {
    1:'DataSets/MetaData1_GT_link_CP.csv',
    2:'DataSets/MetaData2_PeterHorvath_GT_link_CP.csv',
    3:'Dataset/NODIRYET'
}
path_to_GT = switcher_path_to_GT.get(DATASET_TAG)

switcher_eatch_iter = {
    1:False,
    2:True,
    3:True
}
each_iteration = switcher_eatch_iter.get(DATASET_TAG)

# Check if GPU avalaible
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

switcher_inputsize = {
    1: 64,
    2: 64,
    3: 128
}
input_size = switcher_inputsize.get(DATASET_TAG)

#Batch_size will depend on the architecture choice
#Take the biggest batch that can fit in GPU
batch_size = 256

train_loader, valid_loader = get_train_val_dataloader(datadir,input_size,batch_size,test_split=0.15)

zdim=3
base_enc=32
base_dec=32
depth_factor_dec=2
epochs = 150 #Max epoch (early stopping is also implemented)

alpha_range = [0,5,20,100,500,1000]
beta_range = [0,1,5,20,100,500]

DOC_TEXT.append(('batch_size :',batch_size,
                'zdim :', zdim,
                'base_enc :', base_enc,
                'base_dec :', base_dec,
                'max_epochs :', epochs,
                'alpha_range :', alpha_range,
                'beta_range :', beta_range))

with open(save_folder+"Experiment_DOC.txt", "w") as output:
    output.write(str(DOC_TEXT))

counter=1
for alpha in alpha_range:
    for beta in beta_range:
        print('###################################################')
        print(f'Start Model Combination {counter} out of {len(alpha_range)*len(beta_range)}')
        print('###################################################')


        plt.close('all')

        #Create one distinct folder per model
        model_folder = f'{save_folder}alpha_{alpha}_beta_{beta}/'
        if os.path.exists(model_folder):
            shutil.rmtree(model_folder)
        os.makedirs(model_folder)

        VAE = CNN_VAE(zdim=zdim, alpha=alpha, beta=beta, base_enc=base_enc, base_dec=base_dec, depth_factor_dec=depth_factor_dec)
        MLP = MLP_MI_estimator(64*64*3,zdim=3)

        opti_VAE = optim.Adam(VAE.parameters(), lr=0.0001, betas=(0.9, 0.999))
        opti_MLP = optim.Adam(MLP.parameters(), lr=0.0005, betas=(0.9, 0.999))

        if train_on_gpu:
            VAE.cuda()
            MLP.cuda()

        model_name = f'3chan_3z_InfoMAX_a_{alpha}_b_{beta}_{datetime.date.today()}'
        save_model_path = model_folder+f'{model_name}.pth'

        VAE, MLP, history, best_epoch = train_InfoMAX_model(epochs, VAE, MLP, opti_VAE, opti_MLP, train_loader, valid_loader,saving_path=save_model_path,each_iteration=False, train_on_gpu=train_on_gpu)
        fig = plot_train_result(history, best_epoch,save_path=model_folder, infoMAX = True)

        #SAVE TRAINED MODEL and history
        history_save = model_folder+f'loss_evo_{model_name}.csv'
        history.to_csv(history_save)

        #Visualize on the WHOLE dataset (train & validation)
        infer_data, infer_dataloader = get_inference_dataset(datadir,batch_size,input_size,droplast=False)
        #Load the model that has been trained above
        #Where to save csv with metadata
        csv_save_output = model_folder+f'{model_name}_metedata.csv'
        save_csv = True
        #Store raw image data in csv (results in heavy file, but raw data is needed for some metrics)
        store_raw = True

        figplotly = metadata_latent_space(VAE, infer_dataloader, train_on_gpu, GT_csv_path=path_to_GT, save_csv=save_csv, with_rawdata=store_raw,csv_path=csv_save_output)
        html_save = model_folder+f'{model_name}_Representation.html'
        plotly.offline.plot(figplotly, filename=html_save, auto_open=False)

        #save image of reconstruction and generated samples
        image_save = model_folder+f'{model_name}.png'
        save_reconstruction(infer_dataloader,VAE,image_save,train_on_gpu)


        counter+=1

print("#### Full Optimization is terminated")

#31h run
# -> to 23h of run with new MI bounds
