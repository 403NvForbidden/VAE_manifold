# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-03T09:27:00+11:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-04-14T16:53:09+10:00

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
batch_size = 32

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

### ATTENTION CHANGER AUSSI ZDIM THAN TRAIN RANDOM SAMPLE
model = VAE(zdim=512,channels=4,base=16,loss='MSE',layer_count=2)
if train_on_gpu:
    model.cuda()

#print(model)
#summary(model,input_size=(4,256,256),batch_size=32)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 105

model, history = train_VAE_model(epochs, model, optimizer, dataloader, train_on_gpu)

plot_train_result(history)

model_name = '4chan_105e_512z_model2'

#SAVE TRAINED MODEL and history
history_save = 'outputs/plot_history/'+f'loss_evo_{model_name}_{datetime.date.today()}.pkl'
#Save Training history
with open(history_save, 'wb') as f:
    pkl.dump(history, f, protocol=pkl.HIGHEST_PROTOCOL)

save_model_path = 'outputs/saved_models/'+f'VAE_{model_name}_{datetime.date.today()}.pth'
save_checkpoint(model,save_model_path)


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

date = '2020-04-10'
load_model_path = 'outputs/saved_models/'+f'VAE_4chan_3z_40e_{date}.pth'

model = VAE(zdim=3) ## TODO: Modularize that to have the network built inside the load function
if train_on_gpu:
    model.cuda()
model, optimizer = load_checkpoint(model,load_model_path)

# %%
####Continue to train it
epochs = 40

model, history = train_VAE_model(epochs, model, optimizer, dataloader, train_on_gpu)

plot_train_result(history)

# %%
#SEE THE RECONSTRUCTION FOR RANDOM SAMPLE OF VAL DATASET
inference_recon(model, dataloader['val'], 16, train_on_gpu)
