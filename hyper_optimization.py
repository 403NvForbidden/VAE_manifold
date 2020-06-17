# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-06-12T09:10:13+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-06-12T14:30:37+10:00

'''
File running and saving results of several training of same models with
different hyperparameters (in a Grid Search Manner)
'''

import datetime
import os
import errno

import torch
from torchsummary import summary
from torch import cuda, optim

from data_processing import get_dataloader, image_tranforms
from train_net import train_InfoMAX_model, inference_recon
from helpers import plot_train_result, save_brute, load_brute, plot_latent_space, metadata_latent_space
from infoMAX_VAE import CNN_VAE, MLP_MI_estimator


#####################################
#### Define main variable ###########
#####################################
EXPERIMENT_TAG = f'run1_{datetime.date.today()}'
DOC_TEXT = []

MODEL_TAG = 3 # 1: Vanilla 2: SKIP-connection 3: InfoMax 4: InfoMAX + SkipConnection
DATASET_TAG = 1 # 1: BBBC Synthetic Dataset 2: Peter Horvarth Synthetic Dataset 3: Christine Chaffer real Dataset
DESCRIPTION_STR = 'Optimization of encoder and decoder capicity. Objective function is fixed'


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


with open(save_folder+"Experiment_DOC.txt", "w") as output:
    output.write(str(DOC_TEXT))

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

# Check if GPU avalaible
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

switcher_inputsize = {
    1: 64,
    2: 64,
    3: 128
}
input_size = switcher_inputsize.get(DATASET_TAG)
trsfm = image_tranforms(input_size)

#Batch_size will depend on the architecture choice
#Take the biggest batch that can fit in GPU
batch_size = 

_, dataloader = get_dataloader([datadir,'',''],trsfm,batch_size)

zdim=3
alpha=15
beta=1
base_enc=32
base_dec=32
depth_factor_dec=2
epochs = 3

VAE = CNN_VAE(zdim=3, alpha=15, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2)
MLP = MLP_MI_estimator(zdim=3)

opti_VAE = optim.Adam(VAE.parameters(), lr=0.0001, betas=(0.9, 0.999))
opti_MLP = optim.Adam(MLP.parameters(), lr=0.00001, betas=(0.9, 0.999))

if train_on_gpu:
    VAE.cuda()
    MLP.cuda()

summary(VAE,input_size=(3,64,64),batch_size=batch_size)

#VAE, MLP, history = train_InfoMAX_model(epochs, VAE, MLP, opti_VAE, opti_MLP, dataloader, train_on_gpu)
