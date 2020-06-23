# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-04-30T11:43:21+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-06-21T18:03:14+10:00


##########################################################
# %% imports
##########################################################
#%matplotlib qt
from infoMAX_VAE import CNN_VAE, MLP_MI_estimator
from torchsummary import summary
from torch import cuda, optim
from data_processing import get_train_val_dataloader, image_tranforms, imshow_tensor, get_inference_dataset
from train_net import train_InfoMAX_model, inference_recon
from helpers import plot_train_result, save_checkpoint, load_checkpoint, save_brute, load_brute, plot_latent_space, metadata_latent_space, save_reconstruction
import torch
import datetime
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.offline
#plt.ion()

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
batch_size = 256

train_loader, valid_loader = get_train_val_dataloader(datadir_BBBC,input_size,batch_size,test_split=0.15)

#Qualitative inspection of one data example
trainiter = iter(train_loader)
features, labels = next(trainiter)
_,_ = imshow_tensor(features[0])


##########################################################
# %% Build custom VAE Model and TRAIN IT
##########################################################

VAE = CNN_VAE(zdim=3, alpha=15, beta=1, base_enc=32, base_dec=32, depth_factor_dec=2)
MLP = MLP_MI_estimator(zdim=3)


opti_VAE = optim.Adam(VAE.parameters(), lr=0.00005, betas=(0.9, 0.999))
opti_MLP = optim.Adam(MLP.parameters(), lr=0.00005, betas=(0.9, 0.999))

if train_on_gpu:
    VAE.cuda()
    MLP.cuda()

summary(VAE,input_size=(3,64,64),batch_size=32)

#Max number of epochs (can also early stopped if val loss not improve for a long period)
epochs = 500

model_name = f'3chan_dataset1_3z_InfoMAX_{datetime.date.today()}'
save_model_path = 'temporary_save/'+f'{model_name}.pth'

VAE, MLP, history, best_epoch = train_InfoMAX_model(epochs, VAE, MLP, opti_VAE, opti_MLP, train_loader, valid_loader,saving_path=save_model_path,each_iteration=False, train_on_gpu=train_on_gpu)
fig = plot_train_result(history, best_epoch,save_path=None, infoMAX = True)
fig.show()
plt.show()

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
model_VAE = load_brute('temporary_save/3chan_dataset1_3z_InfoMAX_2020-06-19_VAE_.pth')
model_name='3chan_dataset1_3z_InfoMAX_2020-06-19_VAE_'
#Where to save csv with metadata
csv_save_output = 'temporary_save/'+f'{model_name}_metedata.csv'
save_csv = False
#Store raw image data in csv (results in heavy file, but raw data is needed for some metrics)
store_raw = False

figplotly = metadata_latent_space(model_VAE, infer_dataloader, train_on_gpu, GT_csv_path=path_to_GT, save_csv=save_csv, with_rawdata=store_raw,csv_path=csv_save_output)
html_save = 'temporary_save/'+f'{model_name}_Representation.html'
plotly.offline.plot(figplotly, filename=html_save, auto_open=False)

#save image of reconstruction and generated samples
image_save = 'temporary_save/'+f'{model_name}.png'
save_reconstruction(infer_dataloader,model_VAE,image_save,train_on_gpu)

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
# INFERENCE LATENT REPRESENTATION PLOT
batch_size = 128
input_size = 64
dataset_name = 'Synthetic_Data_1'
infer_data, infer_dataloader = get_inference_dataset('DataSets/'+dataset_name,batch_size,input_size,droplast=False)



model_VAE = load_brute('temporary_save/3chan_dataset1_3z_InfoMAX_2020-06-19_VAE_.pth')

#Path to CSV that contains GT of the dataset
#path_to_GT = 'DataSets/MetaData2_PeterHorvath_GT_link_CP.csv'
path_to_GT = 'DataSets/MetaData1_GT_link_CP.csv'
#Where to save csv with metadata
csv_save_output = 'DataSets/John_Metadata_PeterHorvarth_VAELatentCode_20200610.csv'
save_csv = False
#Store raw image data in csv (results in heavy file, but raw data is needed for some metrics)
store_raw = False
figplotly = metadata_latent_space(model_VAE, infer_dataloader, train_on_gpu, GT_csv_path=path_to_GT, save_csv=save_csv, with_rawdata=store_raw,csv_path=csv_save_output)

import plotly.offline
plotly.offline.plot(figplotly, filename='test.html', auto_open=True)


# %%
# Test of SCORE

import plotly.express as px
import plotly.graph_objects as go

import sys, os
import plotly.offline
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication


def show_in_window(fig):


    plotly.offline.plot(fig, filename='name.html', auto_open=False)

    app = QApplication(sys.argv)
    web = QWebEngineView()
    file_path = os.path.abspath(os.path.join(os.getcwd(), "name.html"))
    web.load(QUrl.fromLocalFile(file_path))
    web.show()
    sys.exit(app.exec_())

full_csv = pd.read_csv('DataSets/Sacha_Metadata_2dlatentVAEFAIL_20200518.csv')

#Define where are the source phenotype in latent space
red_cells = full_csv['GT_Shape']<0.15
green_cells = full_csv['GT_Shape']>0.35

#Take the 20 cells closest to the source phenotype, to define a green center and a red center
x_reds = full_csv[red_cells].nsmallest(20,'GT_dist_toMax_phenotype').x_coord.values
y_reds = full_csv[red_cells].nsmallest(20,'GT_dist_toMax_phenotype').y_coord.values
x_greens = full_csv[green_cells].nsmallest(20,'GT_dist_toMax_phenotype').x_coord.values
y_greens = full_csv[green_cells].nsmallest(20,'GT_dist_toMax_phenotype').y_coord.values

red_latent_center = [np.mean(x_reds),np.mean(y_reds)]
green_latent_center = [np.mean(x_greens),np.mean(y_greens)]

trace = go.Scatter(x=[red_latent_center[0],green_latent_center[0]],y=[red_latent_center[1],green_latent_center[1]],
    mode='markers',marker_symbol='x',marker_color='red',
    marker=dict(size=12, opacity=1),
    name=f'Centers')

figplotly.add_traces(trace)

#Define a center of strongest phenotype in each cluster in latent code
cluster_strong_centers = []
cluster_list = np.unique(full_csv.GT_label.values)
for cluster in cluster_list:
    cluster_index = full_csv['GT_label']==cluster
    x_clusters = full_csv[cluster_index].nlargest(15,'GT_dist_toMax_phenotype').x_coord.values
    y_clusters = full_csv[cluster_index].nlargest(15,'GT_dist_toMax_phenotype').y_coord.values
    cluster_strong_centers.append([np.mean(x_clusters),np.mean(y_clusters)])

cluster_strong_centers=np.asarray(cluster_strong_centers)
trace2 = go.Scatter(x=cluster_strong_centers[:6,0],y=cluster_strong_centers[:6,1],
    mode='markers',marker_symbol='x',marker_color='black',
    marker=dict(size=8, opacity=1),
    name=f'Strong Phenotype')

figplotly.add_traces(trace2)

#%%
#Compute distance to max phenotype in VAE latent code

def closest_point(point,points):
    points = np.asarray(points)
    dist_2 = np.sum((points-point)**2, axis=1)
    return np.argmin(dist_2), dist_2[np.argmin(dist_2)]

#Compute and add distance to maximum phenotype per cluster in GT dataframe
distances = []
Extremes = np.array([green_latent_center,red_latent_center])
#Find distance to max Phenotype
for index, row in full_csv.iterrows():
    ind, dist = closest_point(np.array([row['x_coord'],row['y_coord']]),Extremes)
    distances.append(np.sqrt(dist))
full_csv['latent_dist_toMax_phenotype'] = distances

#Normalize to have the distance of center with strong_phenotype center = to 1
cluster_list = np.unique(full_csv.GT_label.values)
for i, cluster in enumerate(cluster_list):
    cluster_index = full_csv['GT_label']==cluster
    ind, normal_dist = closest_point(cluster_strong_centers[i],Extremes)
    full_csv['latent_dist_toMax_phenotype'][cluster_index] = full_csv['latent_dist_toMax_phenotype'][cluster_index].values / np.sqrt(normal_dist)


#full_csv.to_csv('DataSets/Sacha_Metadata_2dlatentVAEFAIL_20200518.csv',index=False)
# %%
full_csv = full_csv.sort_values(by='GT_dist_toMax_phenotype')

#Disregard cluster 7 MSE because no coherent manifold
no_cluster_7 = full_csv['GT_label']!=7

line1 = go.Scatter(y=full_csv[no_cluster_7].GT_dist_toMax_phenotype.values,
    mode='lines',name='GT_distance',line=dict(width=4))
line2 = go.Scatter(y=full_csv[no_cluster_7].latent_dist_toMax_phenotype.values,
    mode='markers',name='Latent_distance',marker=dict(size=3))
layout=go.Layout(title='test')
fig_te = go.Figure(data=[line1,line2],layout=layout)
# fig_3d_1 = go.Figure(data=[go.Scatter3d(x=MetaData_csv.x_coord.values,y=MetaData_csv.y_coord.values,
#     z=MetaData_csv.z_coord.values, mode='markers',
#     marker=dict(size=3,color=MetaData_csv.GT_label.values,
#         opacity=0.8), text=MetaData_csv.GT_Shape.values)])


#%% Calculate a score (MSE) for the distance_to_strong_phenotype fitting
from sklearn import metrics

#Disregard cluster 7 MSE because no coherent manifold
no_cluster_7 = full_csv['GT_label']!=7
GT_distance = full_csv[no_cluster_7].GT_dist_toMax_phenotype.values
latent_distance = full_csv[no_cluster_7].latent_dist_toMax_phenotype.values

overall_mse = metrics.mean_squared_error(GT_distance, latent_distance)

mse_per_cluster = []
cluster_list = np.unique(full_csv.GT_label.values)
for cluster in cluster_list:
    cluster_index = full_csv['GT_label']==cluster
    mse_per_cluster.append(metrics.mean_squared_error(full_csv[cluster_index].GT_dist_toMax_phenotype.values,full_csv[cluster_index].latent_dist_toMax_phenotype.values))

fig_te.update_layout(margin=dict(l=1.1,r=1.1,b=1.1,t=30),showlegend=True,legend=dict(y=-.1),title=dict(text=f'VAE Collapse, Distance to strong phenotype error | MSE : {overall_mse:.4f}'))
fig_te.update_layout(title={'yref':'paper','y':1,'yanchor':'bottom'},title_x=0.5)
fig_te.show()

fig2 = px.bar(x=cluster_list[:6],y=mse_per_cluster[:6])
fig2.update_layout(title='MSE per cluster')
fig2.show()
