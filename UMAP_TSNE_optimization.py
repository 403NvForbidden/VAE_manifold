# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-07-06T09:10:15+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-07-09T17:05:37+10:00

'''
From a CSV file that contains different UMAP (or TSNE) projection, compute all
the latent representation quality metrics to pick the hyperparameter value that
lead to the best result.
'''

import os
import pandas as pd
import numpy as np
from quantitative_metrics.performance_metrics import compute_perf_metrics
from scipy import stats

#########################################################
####### UMAP ####
#########################################################
#Latent codes are saved in columns named UMAP_*NUM*_X/Y/Z

# path_to_csv = 'UMAP-tSNE/20200706_UMAP_dataset1/20200630_UMAP_vs_VAE/200624_Horvath_Simple_DS_with_tSNE_UMAP_VAE-UMAP.csv'
#
# data_df = pd.read_csv(path_to_csv)
# #Extract the UMAP latent code columns
# umap_df = data_df.filter(regex='^UMAP',axis=1).filter(regex='[XYZ]$',axis=1)
#
# col_names = umap_df.columns
# param_values = [name.split('_')[1] for name in col_names]
# param_values = np.unique(param_values)
# param_values
#
#
# params_preferences = {
#     'feature_size':64*64*3,
#     'path_to_raw_data':'DataSets/Synthetic_Data_1',
#     'low_dim_names':'',
#
#     'global_saving_path':'UMAP-tSNE/20200706_UMAP_dataset1/20200630_UMAP_vs_VAE/',
#
#     ### Unsupervised metrics
#     'save_unsupervised_metric':True,
#     'only_local_Q':False,
#     'kt':300,
#     'ks':500,
#
#     ### Mutual Information
#     'save_mine_metric':True,
#     'batch_size':1024,
#     'bound_type':'infoNCE',
#     'alpha_logit':-2.,
#     'epochs':200,
#
#     ### Classifier accuracy
#     'save_classifier_metric':True,
#     'num_iteration':8,
#
#     ### BackBone Metric
#     'save_backbone_metric':True
# }
#
# counter = 0
# for param in param_values :
#
#     print(f'##########################################################')
#     print(f'### Starting Combination {counter} over {len(param_values)-1}')
#     print(f'##########################################################')
#
#     low_dim_names = [f'UMAP_{param}_X',f'UMAP_{param}_Y',f'UMAP_{param}_Z']
#
#     params_preferences['low_dim_names'] = low_dim_names
#
#     general_path = params_preferences["global_saving_path"]
#     save_folder = f'{general_path}UMAP_{param}/metrics/'
#     try:
#         os.makedirs(save_folder)
#     except FileExistsError as e:
#         pass
#
#     params_preferences["global_saving_path"] = save_folder
#
#     ###########################
#     ### Manage OLD RUN umap
#     ###########################
#
#     if 'GT_dist_toMax_phenotype' in data_df.columns:
#         data_df = data_df.rename(columns={'GT_dist_toMax_phenotype':'GT_dist_toInit_state'})
#         to_GT = 'DataSets/MetaData1_GT_link_CP.csv'
#         GT_df = pd.read_csv(to_GT,usecols=['Unique_ID','GT_initial_state'])
#         data_df = data_df.join(GT_df.set_index('Unique_ID'), on='Unique_ID')
#
#     ###########################
#     ### MANAGE UMAP OUTLIERS
#     ###########################
#
#     data_ol_df = data_df[(np.abs(stats.zscore(data_df[low_dim_names]))<4).all(axis=1)]
#     print('Outliers are disregarded ! (It benefits UMAP)')
#
#     #########################
#     #### Compute metrics
#     #########################
#
#     compute_perf_metrics(data_ol_df,params_preferences=params_preferences)
#
#     params_preferences["global_saving_path"] = general_path
#
#     counter +=1


#########################################################
####### t-SNE ####
#########################################################
#Latent codes are saved in columns named UMAP_*NUM*_X/Y/Z

path_to_csv = 'UMAP-tSNE/20200706_UMAP_dataset1/20200630_UMAP_vs_VAE/200624_Horvath_Simple_DS_with_tSNE_UMAP_VAE-UMAP.csv'

data_df = pd.read_csv(path_to_csv)
#Extract the UMAP latent code columns
umap_df = data_df.filter(regex='^RTSNE',axis=1).filter(regex='[XYZ]$',axis=1)

col_names = umap_df.columns
param_values = [name.split('_')[1] for name in col_names]
param_values = np.unique(param_values)
param_values


params_preferences = {
    'feature_size':64*64*3,
    'path_to_raw_data':'DataSets/Synthetic_Data_1',
    'low_dim_names':'',

    'global_saving_path':'UMAP-tSNE/20200706_UMAP_dataset1/tSNE_test/',

    ### Unsupervised metrics
    'save_unsupervised_metric':False,
    'only_local_Q':False,
    'kt':300,
    'ks':500,

    ### Mutual Information
    'save_mine_metric':True,
    'batch_size':256,
    'bound_type':'interpolated',
    'alpha_logit':-4.6,
    'epochs':1000,

    ### Classifier accuracy
    'save_classifier_metric':False,
    'num_iteration':8,

    ### BackBone Metric
    'save_backbone_metric':False
}

counter = 0
for param in param_values :

    print(f'##########################################################')
    print(f'### Starting Combination {counter} over {len(param_values)-1}')
    print(f'##########################################################')

    low_dim_names = [f'RTSNE_{param}_X',f'RTSNE_{param}_Y',f'RTSNE_{param}_Z']

    params_preferences['low_dim_names'] = low_dim_names

    general_path = params_preferences["global_saving_path"]
    save_folder = f'{general_path}RTSNE_{param}/metrics/'
    try:
        os.makedirs(save_folder)
    except FileExistsError as e:
        pass

    params_preferences["global_saving_path"] = save_folder

    ###########################
    ### Manage OLD RUN umap
    ###########################

    if 'GT_dist_toMax_phenotype' in data_df.columns:
        data_df = data_df.rename(columns={'GT_dist_toMax_phenotype':'GT_dist_toInit_state'})
        to_GT = 'DataSets/MetaData1_GT_link_CP.csv'
        GT_df = pd.read_csv(to_GT,usecols=['Unique_ID','GT_initial_state'])
        data_df = data_df.join(GT_df.set_index('Unique_ID'), on='Unique_ID')

    ###########################
    ### MANAGE UMAP OUTLIERS
    ###########################

    data_ol_df = data_df[(np.abs(stats.zscore(data_df[low_dim_names]))<4).all(axis=1)]
    print('Outliers are disregarded ! (It benefits UMAP)')

    #########################
    #### Compute metrics
    #########################

    compute_perf_metrics(data_ol_df,params_preferences=params_preferences)

    params_preferences["global_saving_path"] = general_path

    counter +=1
