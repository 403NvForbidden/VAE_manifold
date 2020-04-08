# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-03T09:27:00+11:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahaidinger
# @Last modified time: 2020-04-07T17:58:04+10:00

##########################################################
# %% imports
##########################################################

from network import VAE
from torchsummary import summary
from torch import cuda, optim
from data_processing import get_dataloader, image_tranforms, imshow_tensor
from train_net import train_VAE_model, plot_train_result, save_checkpoint, load_checkpoint, inference_recon
import torch
import datetime
import pickle as pkl

##########################################################
# %% Define variable
##########################################################
# Location of data
datadir = 'datadir/'
traindir = datadir + 'train/'
validdir = datadir + 'val/'
testdir = datadir + 'test/'
# Change to fit hardware
batch_size = 256

# Check if GPU avalaible
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')


##########################################################
# %% DataLoader and Co
##########################################################
trsfm = image_tranforms()

_, dataloader = get_dataloader([traindir,validdir,testdir],trsfm,batch_size)

trainiter = iter(dataloader['train'])
features, labels = next(trainiter)

_,_ = imshow_tensor(features[0])


##########################################################
# %% Build custom VAE Model
##########################################################

model = VAE(zdim=3)
if train_on_gpu:
    model.cuda()

#print(model)
summary(model,input_size=(4,128,128),batch_size=128)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 500

model, history = train_VAE_model(epochs, model, optimizer, dataloader, train_on_gpu)

plot_train_result(history)

#SAVE TRAINED MODEL and history
history_save = 'outputs/plot_history/'+f'loss_evo_toytrain_{datetime.date.today()}.pkl'
#Save Training history
with open(history_save, 'wb') as f:
    pkl.dump(history, f, protocol=pkl.HIGHEST_PROTOCOL)

save_model_path = 'outputs/saved_models/'+f'VAE_toytrain_{datetime.date.today()}.pth'
save_checkpoint(model,save_model_path)


##########################################################
# %% Visualize training history
##########################################################

history_load = 'outputs/plot_history/'+f'loss_evo_toytrain_2020-04-06.pkl'
with open(history_load, 'rb') as f:
    history = pkl.load(f)

plot_train_result(history)



##########################################################
# %% Load an existing model and continue to train it (or make pred)
##########################################################

date = '2020-04-09'
load_model_path = 'outputs/saved_models/'+f'VAE_toytrain_{date}.pth'

model = VAE(zdim=3) ## TODO: Modularize that to have the network built inside the load function
if train_on_gpu:
    model.cuda()
model, optimizer = load_checkpoint(model,load_model_path)

# %%
#SEE THE RECONSTRUCTION FOR RANDOM SAMPLE OF VAL DATASET
inference_recon(model, dataloader['train'], 16, train_on_gpu)
