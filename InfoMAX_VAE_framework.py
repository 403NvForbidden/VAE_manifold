# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-04-30T11:43:21+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-05-13T17:18:21+10:00


##########################################################
# %% imports
##########################################################
#%matplotlib qt
from infoMAX_VAE import CNN_VAE, MLP_MI_estimator
from torchsummary import summary
from torch import cuda, optim
from data_processing import get_dataloader, image_tranforms, imshow_tensor, get_inference_dataset
from train_net import train_InfoMAX_model, inference_recon
from helpers import plot_train_result, save_checkpoint, load_checkpoint, save_brute, load_brute, plot_latent_space, metadata_latent_space
import torch
import datetime
import pickle as pkl
import matplotlib.pyplot as plt
#plt.ion()

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


##########################################################
# %% Build custom VAE Model
##########################################################

VAE = CNN_VAE(zdim=3, alpha=20, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2)
MLP = MLP_MI_estimator(zdim=3)

opti_VAE = optim.Adam(VAE.parameters(), lr=0.001, betas=(0.9, 0.999))
opti_MLP = optim.Adam(MLP.parameters(), lr=0.0001, betas=(0.9, 0.999))

if train_on_gpu:
    VAE.cuda()
    MLP.cuda()

summary(VAE,input_size=(3,64,64),batch_size=32)

epochs = 80


VAE, MLP, history = train_InfoMAX_model(epochs, VAE, MLP, opti_VAE, opti_MLP, dataloader, train_on_gpu)

fig = plot_train_result(history, infoMAX = True, only_train_data=True)
fig.show()
plt.show()
#model_name = '4chan_105e_512z_model2'
model_name = '3chan_dataset1_80e_3z_run4fdgfd_MLP'

#SAVE TRAINED MODEL and history
#history_save = 'outputs/plot_history/'+f'loss_evo_{model_name}_{datetime.date.today()}.pkl'
#Save Training history
#with open(history_save, 'wb') as f:
    #pkl.dump(history, f, protocol=pkl.HIGHEST_PROTOCOL)

save_model_path = 'outputs/saved_models/'+f'VAE_{model_name}_{datetime.date.today()}.pth'
#save_checkpoint(model,save_model_path)
save_brute(MLP,save_model_path)


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
#
date = '2020-04-17'
model_name = 'testMODEL_z2_e300'

load_model_path = 'outputs/saved_models/'+f'VAE_{model_name}_{date}.pth'


#%% INFERENCE LATENT REPRESENTATION PLOT
batch_size = 128
input_size = 64
infer_data, infer_dataloader = get_inference_dataset('DataSets/Synthetic_Data_1',batch_size,input_size)
infer_iter = iter(infer_dataloader)
features, labels, file_names = next(infer_iter)



model_VAE = load_brute('outputs/saved_models/VAE_3chan_dataset1_80e_3z_run4fdgfd_VAEInfo_2020-05-12.pth')

fig, ax, fig2, ax2 = metadata_latent_space(model_VAE, infer_dataloader, train_on_gpu)
ax.set_title('Latent Representation - Label by GT cluster')
#ax2.set_title('Latent Representation - Label by Shape Factor')

#fig2.savefig('LatentRePresentation2.png')

# %%
# PLOT THE LATENT SPACE OF THE Model
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
fig = plot_latent_space(model,dataloader['train'],train_on_gpu)
plt.show(block=True)
