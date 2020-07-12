# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-07-11T21:54:25+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-07-13T08:56:11+10:00



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
from train_net import train_VAE_model, inference_recon
from helpers import plot_train_result, save_brute, load_brute, plot_from_csv, metadata_latent_space, save_reconstruction
from networks import VAE, Skip_VAE
import plotly.offline
import matplotlib.pyplot as plt
import gc
#####################################
#### Define main variable ###########
#####################################
EXPERIMENT_TAG = f'run1_Vanilla_capacity_{datetime.date.today()}'
DOC_TEXT = []

MODEL_TAG = 1 # 1: Vanilla 2: SKIP-connection 3: InfoMax 4: InfoMAX + SkipConnection
DATASET_TAG = 1 # 1: BBBC Synthetic Dataset 2: Peter Horvarth Synthetic Dataset 3: Christine Chaffer real Dataset
DESCRIPTION_STR = '2D Grid search over encoder and decoder capacity. Beta is fixed at 1'


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
global_save_folder = f'optimization/{switcher_model.get(MODEL_TAG)}/{switcher_dataset.get(DATASET_TAG)}/'
try:
    os.mkdir(global_save_folder+EXPERIMENT_TAG)
    global_save_folder = global_save_folder+EXPERIMENT_TAG+'/'
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

switcher_num_class = {
    1:7,
    2:8,
    3:10
}
num_class = switcher_num_class.get(DATASET_TAG)

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
batch_size_init = 256

zdim=3
beta=1
base_enc=32
base_dec=32
depth_factor_dec=2
epochs = 200 #Max epoch (early stopping is also implemented)

#alpha_range = [0,5,20,100,500,1000]
#beta_range = [0,1,5,20,100,500]
enc_range = [4,8,16,32,64,128]
dec_range = [4,8,16,32,64,128]


DOC_TEXT.append(('batch_size :',batch_size_init,
                'zdim :', zdim,
                #'base_enc :', base_enc,
                #'base_dec :', base_dec,
                'beta :', beta,
                'max_epochs :', epochs,
                'enc_range :', enc_range,
                'dec_range :', dec_range))

with open(global_save_folder+"Experiment_DOC.txt", "w") as output:
    output.write(str(DOC_TEXT))

counter=1
for enc in enc_range:
    for dec in dec_range:
        print('###################################################')
        print(f'Start Model Combination {counter} out of {len(enc_range)*len(enc_range)}')
        print('###################################################')
        batch_size = batch_size_init
        ### Batch size might lead to full GPU if capacity increase, manage that case

        #Create one distinct folder per model
        model_folder = f'{global_save_folder}/models/enc_{enc}_dec_{dec}/'
        if os.path.exists(model_folder):
            shutil.rmtree(model_folder)
        os.makedirs(model_folder)

        save_folder = model_folder+'model_training/'
        os.makedirs(save_folder)


        while batch_size >= 64: #Ensure that the batch size is at least of size 64
            try:
                batch_size = batch_size
                train_loader, valid_loader = get_train_val_dataloader(datadir,input_size,batch_size,test_split=0.15)

                model = VAE(zdim=zdim, beta=beta, base_enc=enc, base_dec=dec, depth_factor_dec=depth_factor_dec)
                optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

                if train_on_gpu:
                    model.cuda()

                model_name = f'3chan_bs{batch_size}_vanilla_enc_{enc}_dec_{dec}_{datetime.date.today()}'
                save_model_path = save_folder+f'{model_name}.pth'

                model, history, best_epoch = train_VAE_model(epochs, model, optimizer, train_loader, valid_loader,saving_path=save_model_path,each_iteration=False, train_on_gpu=train_on_gpu)

                break #Enough memory on GPU with the fixed batch_size, leave the while loop

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    batch_size = int(batch_size/2.)
                    print(f'Batch Size is too big - reduce it to {batch_size}')
                    if batch_size < 64:
                        print('CAUTION : Batch Size 64 is still to heavy, nothing is saved for this architecture')
                        break
                    with torch.no_grad():

                        del model
                        del optimizer
                        gc.collect()
                        torch.cuda.empty_cache()
                    continue

        if batch_size >= 64:
            fig = plot_train_result(history, best_epoch,save_path=save_folder, infoMAX = False)

            #SAVE TRAINED MODEL and history
            history_save = save_folder+f'loss_evo_{model_name}.csv'
            history.to_csv(history_save)

            #Visualize on the WHOLE dataset (train & validation)
            infer_data, infer_dataloader = get_inference_dataset(datadir,batch_size,input_size,droplast=False)
            #Load the model that has been trained above
            #Where to save csv with metadata
            csv_save_output = model_folder+f'{model_name}training_metadata.csv'
            save_csv = True
            #Store raw image data in csv (results in heavy file)
            store_raw = False

            #Save csv file that link latent code to ground truth infomartion
            MetaData_csv = metadata_latent_space(model, infer_dataloader, train_on_gpu, GT_csv_path=path_to_GT, save_csv=save_csv, with_rawdata=store_raw,csv_path=csv_save_output)

            #Save 3D interative plot
            figplotly = plot_from_csv(MetaData_csv,dim=3,num_class=num_class)
            html_save = save_folder+f'{model_name}_Representation.html'
            plotly.offline.plot(figplotly, filename=html_save, auto_open=False)

            #save image of reconstruction and generated samples
            image_save = save_folder+f'{model_name}.png'
            save_reconstruction(infer_dataloader,model,image_save,train_on_gpu)
            plt.close('all')

            counter+=1

print("#### Full Optimization is terminated")


#Run of 29h23m
