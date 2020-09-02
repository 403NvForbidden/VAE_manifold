# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-08-19T09:29:45+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-31T12:51:57+10:00

'''
Playground to train a VAE on a toy dataset, a fine tune it with human feedback
Human feedback is given to VAE based on an additional term in its objective function

This work is meant to be a 'proof of concept'; Feedback can be incorporated in VAE weights as long
as it can be translated as an additionnal term inside the objective function.
We studied here a very simple case with Google Dsprites dataset.
The feedback is in the form of anchored several points to a given region of the latent space.
That is perform with a MSE addtional term, weighted by a parameters that is sample depended, meaning
that it is set to 0 for all samples except the one that are constrained. This information need to be
stored in a CSV file.

To have more information on how the CSV file should be built, please refer to the class DSpritesDataset in feedback_helpers.py

More information on the dataset : https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb
DSprites Dataset from [1].
----------
[1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick,
    M., ... & Lerchner, A. (2017). beta-vae: Learning basic visual concepts
    with a constrained variational framework. In International Conference
    on Learning Representations.
'''

import numpy as np
import pandas as pd
import datetime
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch import cuda, optim
from torch.autograd import Variable

from util.helpers import save_brute, load_brute
from models.train_net import train_Simple_VAE
from models.networks import Simple_VAE
from util.data_processing import imshow_tensor
from human_guidance.feedback_helpers import latent_to_index, sample_latent, show_images_grid, show_density, DSpritesDataset, save_latent_code, plot_train_history

###############################################
### Load DSprites Dataset and Inspect it
###############################################
path_to_dsprites = 'human_guidance/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
full_dataset = np.load(path_to_dsprites)

full_dataset.files
imgs = full_dataset['imgs']
GT_factors = full_dataset['latents_values']
GT_class = full_dataset['latents_classes']
#Associate a unique ID with each datasample
Unique_index = np.arange(imgs.shape[0])

# Sample latents randomly
latents_sampled = sample_latent(size=5000)
# Select images
indices_sampled = latent_to_index(latents_sampled)
imgs_sampled = imgs[indices_sampled]
# Show images
show_images_grid(imgs_sampled)

#%%
#################################
### Create Subsample with desired properties
#################################
#To do so, use the latent_to_index method
# The 6 latent factors of variations are in the following order :
# 0 1 2 3 4 5
# ('color', 'shape', 'scale', 'orientation', 'posX', 'posY')
# To understand better, visit https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb

### Select all square, all scale, all X-Y pos, but NO rotation
shape_mask = GT_class[:,1]==0 #Select only square shape
rotation_mask = [GT_class[i,3] in [0] for i in range(GT_class.shape[0])] #Select only no rotation
final_mask = [np.all(tup) for tup in zip(shape_mask,rotation_mask)]
sub_sample1 = GT_class[final_mask]

############ STUDY CASE 1 ###############
### select few heart, 2 scale, 16 sparse X-Y position, orientation fixed
# shape_mask2 = GT_class[:,1]==2
# scale_mask2 = [GT_class[i,2] in [3,4] for i in range(GT_class.shape[0])]
# posx_mask2 = [GT_class[i,4] in [5,10,15,25] for i in range(GT_class.shape[0])]
# posy_mask2 = [GT_class[i,5] in [5,15,20,25] for i in range(GT_class.shape[0])]
# rotation_mask2 = [GT_class[i,3] in [0] for i in range(GT_class.shape[0])]
# final_mask2= [np.all(tup) for tup in zip(shape_mask2,posx_mask2,posy_mask2,rotation_mask2,scale_mask2)]
# sub_sample2 = GT_class[final_mask2]

############ STUDY CASE 2 ###############
### Select all heart, all scale, all X-Y pos, but NO rotation
shape_mask2 = GT_class[:,1]==2 #Select only square shape
rotation_mask2 = [GT_class[i,3] in [0] for i in range(GT_class.shape[0])] #Select only no rotation
final_mask2 = [np.all(tup) for tup in zip(shape_mask2,rotation_mask2)]
sub_sample2 = GT_class[final_mask2]
sub_sample2.shape


sub_sample=np.concatenate((sub_sample1,sub_sample2))
indices_sampled = latent_to_index(sub_sample)


imgs_sampled = imgs[indices_sampled]
GT_factors_sampled = GT_factors[indices_sampled]
GT_class_sampled = GT_class[indices_sampled]
Unique_index_sampled = Unique_index[indices_sampled]

#Create first Train Dataloader with no feedback yet
dataLoader = DataLoader(DSpritesDataset(imgs_sampled,GT_factors_sampled,Unique_index_sampled,feedback_csv='None'),batch_size=256,shuffle=True)

#Qualitative inspection of one data example
trainiter = iter(dataLoader)
features, labels, id, feedbacks = next(trainiter)
image = features[0].numpy().transpose((1, 2, 0))
plt.imshow(np.squeeze(image))


#################################################
####### Train VAE with custom dataset - NO FEEDBACK yet
#################################################
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')
model = Simple_VAE(zdim=2, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2)
if train_on_gpu:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
epochs = 3

model_name = f'NoFeedbackYet_{datetime.date.today()}'
save_model_path = f'{model_name}.pth'
model, history = train_Simple_VAE(epochs, model, optimizer,dataLoader, train_on_gpu=train_on_gpu)
save_brute(model,save_model_path)

fig = plot_train_history(history)
fig.show()

#Save latent code and unique index
df = save_latent_code(model,dataLoader,save_path='noname_metadata.csv')

#%%
#########################
#### Plot the latent space before feedback
#########################

label = 'scale' #GT factor to color-code
#Me[column]=MetaData_csv[column].astype(str)
low_dim_names = ['VAE_x_coord','VAE_y_coord']
fig = px.scatter(df,x=low_dim_names[0],y=low_dim_names[1],text='Unique_id',color=label,color_discrete_sequence=px.colors.qualitative.T10+px.colors.qualitative.Alphabet)
fig.update_traces(marker=dict(size=10))
html_save = f'{model_name}_scale_Representation.html'
plotly.offline.plot(fig, filename=html_save, auto_open=True)

#%%
#####################################
###Generate Feedback CSV
#####################################
#From the saved CSV of the initial training, add column for feedbacks :
# This csv file should
#         contain at least 4 columns : 'Unique_id' the index of each sample of the considered dataset,
#         'delta' the weight of the feedback term in objective function (0 if point is not constrained),
#         'x_anchor' x coordinate of the anchor point for each sample,
#         'y_anchor' y coordinate of the anchor point for each sample

#CSV of the initial training
df = pd.read_csv('noname_metadata.csv')
df['delta']=0
df['x_anchor']=0
df['y_anchor']=0
anchored_points = df['shape']==3
df.loc[anchored_points,'x_anchor']=-1
df.loc[anchored_points,'y_anchor']=0
df.loc[anchored_points,'delta']=0.005
anchored_points2 = df['shape']==1
df.loc[anchored_points2,'x_anchor']=1
df.loc[anchored_points2,'y_anchor']=0
df.loc[anchored_points2,'delta']=0.005
#Save the CSV with the feedback
df.to_csv(f'noname_wFEEDBACK_metadata.csv')


#%%
###############################
#### Fine Tune with FEEDBACK
###############################
feedback_csv = 'noname_wFEEDBACK_metadata.csv'
dataLoader = DataLoader(DSpritesDataset(imgs_sampled,GT_factors_sampled,Unique_index_sampled,feedback_csv=feedback_csv),batch_size=256,shuffle=True)

train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

#LOAD THE INITIAL TRAINED VAE
model = load_brute(f'NoFeedbackYet_{datetime.date.today()}.pth')
if train_on_gpu:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
epochs = 5
#model.beta = 0.1
model_name = f'VAE_withFeedback_{datetime.date.today()}'
save_model_path = f'{model_name}.pth'

model, history = train_Simple_VAE(epochs, model, optimizer,dataLoader, train_on_gpu=train_on_gpu)
save_brute(model,save_model_path)

fig = plot_train_history(history)
fig.show()

df = save_latent_code(model,dataLoader,save_path='noname_AfterFEEDBACK_metadata.csv')

#%%
##################################
#### Plot Model AFTER FEEDBACK
##################################
label = 'scale'
#Me[column]=MetaData_csv[column].astype(str)
low_dim_names = ['VAE_x_coord','VAE_y_coord']
fig = px.scatter(df,x=low_dim_names[0],y=low_dim_names[1],text='Unique_id',color=label,color_discrete_sequence=px.colors.qualitative.T10+px.colors.qualitative.Alphabet)
fig.update_traces(marker=dict(size=10))
fig.add_scatter(x=[1],y=[0])
html_save = f'{model_name}_scale_Representation.html'
plotly.offline.plot(fig, filename=html_save, auto_open=True)

#%%
#For Study CASE 1
############################################
#### Select new UNSEEN sample ######
############################################
#Slect new heart sample that have never been seen during training
shape_mask3 = GT_class[:,1]==2
scale_mask3 = [GT_class[i,2] in [3,4] for i in range(GT_class.shape[0])]
posx_mask3 = [GT_class[i,4] in [17,22] for i in range(GT_class.shape[0])]
posy_mask3 = [GT_class[i,5] in [7,12] for i in range(GT_class.shape[0])]
rotation_mask3 = [GT_class[i,3] in [0] for i in range(GT_class.shape[0])]
final_mask3= [np.all(tup) for tup in zip(shape_mask3,posx_mask3,posy_mask3,rotation_mask3,scale_mask3)]
sub_sample3 = GT_class[final_mask3]
sub_sample3.shape
indices_sampled = latent_to_index(sub_sample3)

imgs_sampled = imgs[indices_sampled]
GT_factors_sampled = GT_factors[indices_sampled]
GT_class_sampled = GT_class[indices_sampled]
Unique_index_sampled = Unique_index[indices_sampled]

#### Check where the VAE project those unseen points before and after the fine tuning with feedback
dataLoader = DataLoader(DSpritesDataset(imgs_sampled,GT_factors_sampled,Unique_index_sampled,feedback_csv='None'),batch_size=6,shuffle=True)
train_on_gpu=True
#model_beforeFeedback = load_brute('Feedbacks/20200820 - No Feedbacks yet/SimpleVAE_run1_and_fewheart2020-08-20.pth')
#model_name = 'BeforeFeedback_UnseenPoint'
model_afterFeedback = load_brute(f'{model_name}.pth')

df = save_latent_code(model_afterFeedback,dataLoader,save_path='noname_UNSEEN_afterFeedback_metadata.csv')

#%%
################################
### Add unseen point to latent space plot
##################################
label = 'shape'
#Me[column]=MetaData_csv[column].astype(str)
low_dim_names = ['VAE_x_coord','VAE_y_coord']
df = pd.read_csv(f'noname_AfterFEEDBACK_metadata.csv')
fig = px.scatter(df,x=low_dim_names[0],y=low_dim_names[1],text='Unique_id',color_discrete_sequence=px.colors.qualitative.T10+px.colors.qualitative.Alphabet)
df2 = pd.read_csv('noname_UNSEEN_afterFeedback_metadata.csv')
fig.update_traces(marker=dict(size=10,opacity=0.7))
fig.add_scatter(x=df2['VAE_x_coord'],y=df2['VAE_y_coord'],mode='markers',marker=dict(size=13))
html_save = f'UNSEEN_point_plot.html'
plotly.offline.plot(fig, filename=html_save, auto_open=True)
