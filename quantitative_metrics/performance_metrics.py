# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-05-25T11:37:48+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-07-08T15:05:31+10:00

'''
From a csv or a pandas DataFrame containing the projection (latent code) of
the dataset, compute and save different performance metrics, that can be used
to evaluate the quality of the learnt representation.
'''

import pandas as pd
import numpy as np
import math
import os
from torch import cuda
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline
import pickle as pkl

from quantitative_metrics.unsupervised_metric import unsup_metric_and_local_Q, save_representation_plot
from quantitative_metrics.MINE_metric import compute_MI
from quantitative_metrics.classifier_metric import classifier_performance
from quantitative_metrics.backbone_metric import dist_preservation_err


### Note, put :
# False  -> to NOT compute a given metric
# 'no_save' -> to compute a metric but NOT save results
# True -> to compute the metric and save results to a given folder
params_preferences = {
    'feature_size':64*64*3,
    'path_to_raw_data':'DataSets/Synthetic_Data_1',
    'low_dim_names':['VAE_x_coord','VAE_y_coord','VAE_z_coord'],

    'global_saving_path':'',

    ### Unsupervised metrics
    'save_unsupervised_metric':True,
    'only_local_Q':False,
    'kt':300,
    'ks':500,

    ### Mutual Information
    'save_mine_metric':True,
    'batch_size':512,
    'bound_type':'infoNCE',
    'aplha_logit':-2.,
    'epochs':400,

    ### Classifier accuracy
    'save_classifier_metric':None,
    'num_iteration':8,

    ### BackBone Metric
    'save_backbone_metric':None
}

def compute_perf_metrics(data_source,params_preferences):
    '''
    From a csv or a pandas DataFrame containing the projection (latent code) of
    the dataset, compute and save different performance metrics, that can be used
    to evaluate the quality of the learnt representation.
    '''

    if isinstance(data_source,str):
        MetaData_df = pd.read_csv(data_source)
    else:
        MetaData_df = data_source

    ##############################################
    ########## Unsupervised Metrics ##############
    ##############################################

    if params_preferences['save_unsupervised_metric'] != False:
        save_path = params_preferences['save_unsupervised_metric']
        if save_path == 'no_save':
            save_path = None
        else:
            save_path = f"{params_preferences['global_saving_path']}unsup_metric"
            try:
                os.mkdir(save_path)
            except FileExistsError as e:
                pass

        print('#######################################')
        print('###### Computing unsupervised score')
        print('#######################################')

        trust_AUC, cont_AUC, lcmc_AUC,light_df = unsup_metric_and_local_Q(MetaData_df,low_dim_names=params_preferences['low_dim_names'],
                    raw_data_included=False, feature_size=params_preferences['feature_size'],
                    path_to_raw_data=params_preferences['path_to_raw_data'], saving_path=save_path,
                    kt=params_preferences['kt'], ks=params_preferences['ks'],
                    only_local_Q=params_preferences['only_local_Q'])

        MetaData_df = light_df

        save_representation_plot(MetaData_df,save_path,low_dim_names=params_preferences['low_dim_names'])

        print(f'Trustworthiness AUC : {trust_AUC}')
        print(f'Continuity AUC : {cont_AUC}')
        print(f'LCMC AUC : {lcmc_AUC}')


    ##############################################
    ########## Mutual Information ##############
    ##############################################

    if params_preferences['save_mine_metric'] != False:
        save_path = params_preferences['save_mine_metric']
        if save_path == 'no_save':
            save_path = None
        else:
            save_path = f"{params_preferences['global_saving_path']}MI_metric"
            try:
                os.mkdir(save_path)
            except FileExistsError as e:
                pass

        print('#######################################')
        print('###### Computing Mutual Information')
        print('#######################################')

        MI_score = compute_MI(MetaData_df,low_dim_names=params_preferences['low_dim_names'],
                    path_to_raw_data=params_preferences['path_to_raw_data'],save_path=save_path,
                    batch_size=params_preferences['batch_size'],alpha_logit=params_preferences['alpha_logit'],
                    bound_type=params_preferences['bound_type'],epochs=params_preferences['epochs'])

        MI_score_df = pd.DataFrame({'MI_score':MI_score},index=[0])
        if save_path != None :
            MI_score_df.to_csv(f'{save_path}/MI_score.csv')

        print(f'Mutual Information : {MI_score}')

    ##############################################
    ########## Classifier accuracy ##############
    ##############################################

    if params_preferences['save_classifier_metric'] != False:
        save_path = params_preferences['save_classifier_metric']
        if save_path == 'no_save':
            save_path = None
        else:
            save_path = f"{params_preferences['global_saving_path']}classifier_metric"
            try:
                os.mkdir(save_path)
            except FileExistsError as e:
                pass

        print('#######################################')
        print('###### Computing Classifier accuracy')
        print('#######################################')

        print('Metric 1...')
        #Metric 1 : Acc on all test single cells except uniform cluster 7
        _, test_accuracies_m1, _ = classifier_performance(MetaData_df,low_dim_names=params_preferences['low_dim_names'],
                    Metrics=[True,False,False],num_iteration=params_preferences['num_iteration'])
        print('Metric 2...')
        #Metric 2 : Acc on all strong phenotypical change test single cells except uniform cluster 7
        _, test_accuracies_m2, _ = classifier_performance(MetaData_df,low_dim_names=params_preferences['low_dim_names'],
                    Metrics=[False,True,False],num_iteration=params_preferences['num_iteration'])
        print('Metric 3...')
        #Metric 3 : Acc on all strong phenotypical change + META_CLUSTER (1&2, 3&4 and 5&6 grouped) test single cells except uniform cluster 7
        _, test_accuracies_m3, _ = classifier_performance(MetaData_df,low_dim_names=params_preferences['low_dim_names'],
                    Metrics=[False,False,True],num_iteration=params_preferences['num_iteration'])

        mean_acc_m1 = np.mean(test_accuracies_m1)
        std_acc_m1 = np.std(test_accuracies_m1)
        mean_acc_m2 = np.mean(test_accuracies_m2)
        std_acc_m2 = np.std(test_accuracies_m2)
        mean_acc_m3 = np.mean(test_accuracies_m3)
        std_acc_m3 = np.std(test_accuracies_m3)

        accuracy_df = pd.DataFrame({'test_accuracies_m1':test_accuracies_m1,'test_accuracies_m2':test_accuracies_m2,
            'test_accuracies_m3':test_accuracies_m3,'mean_acc_m1':mean_acc_m1,
            'std_acc_m1':std_acc_m1,'mean_acc_m2':mean_acc_m2,
            'std_acc_m2':std_acc_m2,'mean_acc_m3':mean_acc_m3,'std_acc_m3':std_acc_m3})
        if save_path != None:
            accuracy_df.to_csv(f'{save_path}/classifier_acc_score.csv')

        print(f'Accuracy m1 : {mean_acc_m1}')
        print(f'Accuracy m2 : {mean_acc_m2}')
        print(f'Accuracy m3 : {mean_acc_m3}')


    ##############################################
    ########## BackBone / Rank Correlation #######
    ##############################################

    if params_preferences['save_backbone_metric'] != False:
        save_path = params_preferences['save_backbone_metric']
        if save_path == 'no_save':
            save_path = None
        else:
            save_path = f"{params_preferences['global_saving_path']}backbone_metric"
            try:
                os.mkdir(save_path)
            except FileExistsError as e:
                pass

        print('#######################################')
        print('###### Computing Backbone Metric')
        print('#######################################')

        _,spearman_r, kendall_r = dist_preservation_err(MetaData_df,low_dim_names=params_preferences['low_dim_names'],
                overwrite_csv=False,save_path=save_path)

        print(f'Spearman Coefficient : {spearman_r}')
        print(f'Kendall Coefficient : {kendall_r}')

    plt.close('all')
    print('Metrics Computation Terminated')








# # %%
# #
# #
# # #Load the appropriate CSV file
# name_of_csv1 = 'DataSets/Sacha_Metadata_3dlatentVAE_20200523.csv'
# # name_of_csv2 = 'DataSets/Sacha_Metadata_3dlatentVAEFAIL_20200524.csv'
# # name_of_csv3 = 'DataSets/Sacha_Metadata_3dlatentVAEbigFAIL_20200525.csv'
# umap_CSV = 'UMAP-tSNE/20200630_UMAP_vs_VAE/UMAP_61_X_light_metadata.csv'
# low_dim_names = ['UMAP_61_X','UMAP_61_Y','UMAP_61_Z']
#
#
# compute_MI(name_of_csv1,path_to_raw_data='DataSets/Synthetic_Data_1',save_path=None,batch_size=1024,alpha_logit=-5.,bound_type='infoNCE',epochs=150)
# #%%
#
#
# params_preferences = {
#     'feature_size':64*64*3,
#     'path_to_raw_data':'DataSets/Synthetic_Data_1',
#     'low_dim_names':['UMAP_61_X','UMAP_61_Y','UMAP_61_Z'],
#
#     'global_saving_path':'temporary_save/test_folder/',
#
#     ### Unsupervised metrics
#     'save_unsupervised_metric':True,
#     'only_local_Q':False,
#     'kt':300,
#     'ks':500,
#
#     ### Mutual Information
#     'save_mine_metric':True,
#     'batch_size':512,
#     'bound_type':'interpolated',
#     'alpha_logit':-2.,
#     'epochs':20,
#
#     ### Classifier accuracy
#     'save_classifier_metric':True,
#     'num_iteration':4,
#
#     ### BackBone Metric
#     'save_backbone_metric':True
# }
#
#
# compute_perf_metrics(umap_CSV,params_preferences=params_preferences)


#%%

# #MANAGE POSSIBLE OUTLIERS OF UMAP
# umap_df = pd.read_csv(umap_CSV)
# len(umap_df)
# umap_df = umap_df[(np.abs(stats.zscore(umap_df[low_dim_names]))<4).all(axis=1)]
# len(umap_df)
# #
# # #compare_models([name_of_csv1,name_of_csv2,name_of_csv3],Metrics=[False,False,True],num_iteration=10)
# #
# backbone,spearman_r, kendall_r, figplotly = dist_preservation_err(umap_df,low_dim_names,with_plot=True,save_result=False)
# plotly.offline.plot(figplotly, filename='test.html', auto_open=True)
# mse
#
# #
# #
# MI_score = compute_MI(name_of_csv1,save_path=None,batch_size=1024,alpha_logit=-2.0,bound_type='interpolated')
# #MI_score = compute_MI(umap_CSV,low_dim_names=['UMAP_61_X','UMAP_61_Y','UMAP_61_Z'],save_path=None,batch_size=512,alpha_logit=-2.0,bound_type='interpolated')

# %%

#
#
#
#
#
#
# # %%
# csvfile = '../Data_Horvath/200614_Horvath Synth Data complex set Manifolds.csv'
# plotly.colors.qualitative.Plotly[0]
# data = pd.read_csv(csvfile)
# #data['TSNE_60_clusters'] = data['TSNE_60_clusters'].astype(str)
# #without_NC = data['UMAP_23_clusters'] != 'non-clustered'
# data['GT_label'] = data['GT_label'].astype(str)
# fig = px.scatter_3d(data,x='x_coord',y='y_coord',z='z_coord',color='GT_label')
# #plotly.offline.plot(fig, filename = 'VAE.html', auto_open=True)
#
#
#
