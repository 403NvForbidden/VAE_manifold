# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-03T09:27:00+11:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-05-15T23:42:46+10:00

##########################################################
# %% imports
##########################################################

from networks import VAE, Skip_VAE
from torchsummary import summary
from torch import cuda, optim
from data_processing import get_dataloader, image_tranforms, imshow_tensor, get_inference_dataset
from train_net import train_VAE_model, inference_recon
from helpers import plot_train_result, save_checkpoint, load_checkpoint, save_brute, load_brute, plot_latent_space
import torch
import datetime
import pickle as pkl

##########################################################
# %% Define variable
##########################################################
# Location of data
datadir = 'datadir/'
datadir1 = 'DataSets/'
#traindir = datadir + 'train/'
traindir = datadir1 + 'Synthetic_Data_1'
validdir = datadir + 'val/'
testdir = datadir + 'test/'
# Change to fit hardware
batch_size = 128

# Check if GPU avalaible
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')


##########################################################
# %% DataLoader and Co
##########################################################

#Define the input size of the image
# Data will be reshape   C x input_size x input_size

input_size = 64

trsfm = image_tranforms(input_size)
_, dataloader = get_dataloader([traindir,validdir,testdir],trsfm,batch_size)

trainiter = iter(dataloader['train'])
features, labels = next(trainiter)

_,_ = imshow_tensor(features[0])

#%% Test inference DataSets
# infer_data, infer_dataloader = get_inference_dataset('DataSets/Synthetic_Data_1',batch_size,input_size)
# infer_iter = iter(infer_dataloader)
# features, labels, file_names = next(infer_iter)
# print(file_names)
# _,_ = imshow_tensor(features[0])
#
#
# temp_id = [[file_name.split('_')[2],file_name.split('_')[3][-1],file_name.split('_')[4][2:-5]] for file_name in file_names]
# temp_id2 = temp_id
#
# temp = [temp_id, temp_id2]
# print(temp)
# import itertools
# print(list(itertools.chain.from_iterable(temp)))
#
# #Obtain the class
# print(infer_data.classes[0])

##########################################################
# %% Build custom VAE Model
##########################################################


model = VAE(zdim=3, beta=1, base_enc=64, base_dec=32, depth_factor_dec=2)
#model = Skip_VAE(zdim=3, beta=1, base_enc=64, base_dec=32, depth_factor_dec=2)
if train_on_gpu:
    model.cuda()

#print(model)
summary(model,input_size=(3,input_size,input_size),batch_size=32)

optimizer = optim.Adam(model.parameters(), lr=0.5e-4)

epochs = 60

# KL annealing parameter : [max_value, start_increace, reach_max_value]
#beta_init = [1.0,5,55]
model, history = train_VAE_model(epochs, model, optimizer, dataloader, train_on_gpu)

fig = plot_train_result(history, only_train_data=True)
fig.show()

#model_name = '4chan_105e_512z_model2'
#model_name = '3chan_60e_3z_SotaVAE'

#SAVE TRAINED MODEL and history
#history_save = 'outputs/plot_history/'+f'loss_evo_{model_name}_{datetime.date.today()}.pkl'
#Save Training history
#with open(history_save, 'wb') as f:
    #pkl.dump(history, f, protocol=pkl.HIGHEST_PROTOCOL)

#save_model_path = 'outputs/saved_models/'+f'VAE_{model_name}_{datetime.date.today()}.pth'
#save_checkpoint(model,save_model_path)
#save_brute(model,save_model_path)


##########################################################
# %% Visualize training history
##########################################################

history_load = 'outputs/plot_history/'+f'loss_evo_{model_name}_{datetime.date.today()}.pkl'
with open(history_load, 'rb') as f:
    history = pkl.load(f)

plot_train_result(history)



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
