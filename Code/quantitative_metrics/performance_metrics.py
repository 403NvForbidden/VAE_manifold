# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-05-25T11:37:48+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-31T11:19:20+10:00

'''
Man function to compute ALL the performance metrics to assess the quality a learnt representation

From a csv or a pandas DataFrame containing the projection (latent code) of
the dataset, compute and save different performance metrics, that can be used
to evaluate the quality of the learnt representation.

1) Unsupervised Metrics : Trustworthiness, Continuity, LCMC and Local Quality Score that assess neighborhood preservation
2) Mutual Information : Estimation of the Mutual Information between input and latent codes, estimate with a variational lower bound on MI parameterized by a Neural Network (MINE Framework)
3) Classifier accuracy : Gauge the ability of the method to extract features enabling to discriminate bewtween ground truth clusters
4) Backbone metric : Assess ground truth for continuity; how an expected smooth continuum is projected as a trajectory in the latent space

All the important parameters need to be given as a dictionnary name 'params_preferences' :

### Note, put :
# For bolean values :
# False  -> to NOT compute a given metric
# 'no_save' -> to compute a metric but NOT save results
# True -> to compute the metric and save results to a given folder

# params_preferences = {
#     'feature_size':64*64*3,  ### Dimensionality of input data (number of pixels)
#     'path_to_raw_data':'DataSets/Synthetic_Data_1',  ### Path to the folder where raw data (single cell images) are stored
#     'dataset_tag':1, # 1:BBBC 2:Horvath 3:Chaffer
#     'low_dim_names':['VAE_x_coord','VAE_y_coord','VAE_z_coord'], ### name of the columns that stores the latent codes in the main csv file
#
#     'global_saving_path':'path to folder', ### Path to folder where to store the results
#
#     ### Unsupervised metrics
#     'save_unsupervised_metric':True,
#     'only_local_Q':False,  ### If True, compute only the local quality score (save computational time)
#     'kt':300,  ### Neighborhood size parameter kt (refer to local_quality.py)
#     'ks':500,  ### Neighborhood size parameter ks (refer to local_quality.py)
#
#     ### Mutual Information
#     'save_mine_metric':True,
#     'batch_size':512,
#     'bound_type':'infoNCE', ### Variational Bound to use as estimator of Mutual Information
#     'alpha_logit':-2., ### If interpolated bound is use, value of the parameter that control the bias-variance trade-off
#     'epochs':400,
#
#     ### Classifier accuracy
#     'save_classifier_metric':False,
#     'num_iteration':8, # The whole process (new model, new train-test split, new training) will be performed num_iteration time to obtain a mean and std

#
#     ### BackBone Metric
#     'save_backbone_metric':False
# }

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
from util.helpers import plot_from_csv


from quantitative_metrics.unsupervised_metric import unsup_metric_and_local_Q, save_representation_plot
from quantitative_metrics.MINE_metric import compute_MI
from quantitative_metrics.classifier_metric import classifier_performance
from quantitative_metrics.backbone_metric import dist_preservation_err


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

        #MetaData_df = light_df

        save_representation_plot(light_df,save_path,low_dim_names=params_preferences['low_dim_names'])

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

        #BBBC dataset has 3 different classifier metrics.
        if params_preferences['dataset_tag']==1:

            # if 'GT_dist_toMax_phenotype' in MetaData_df.columns:
            #     MetaData_df = MetaData_df.rename(columns={'GT_dist_toMax_phenotype':'GT_dist_toInit_state'})
            #     to_GT = 'DataSets/MetaData1_GT_link_CP.csv'
            #     GT_df = pd.read_csv(to_GT,usecols=['Unique_ID','GT_initial_state'])
            #     data_df = MetaData_df.join(GT_df.set_index('Unique_ID'), on='Unique_ID')

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

        else:

            print('Metric 1...')
            if params_preferences['dataset_tag']==2: #Horvath dataset, imbalanced and 6 class
                _, test_accuracies_m1, _ = classifier_performance(MetaData_df,low_dim_names=params_preferences['low_dim_names'],
                        Metrics=[True,False,False],num_iteration=params_preferences['num_iteration'],
                        num_class=6,class_to_ignore='None',imbalanced_data=True)
            elif params_preferences['dataset_tag']==3: #Chaffer dataset,
                _, test_accuracies_m1, _ = classifier_performance(MetaData_df,low_dim_names=params_preferences['low_dim_names'],
                        Metrics=[True,False,False],num_iteration=params_preferences['num_iteration'],
                        num_class=12,class_to_ignore='None',imbalanced_data=True)

            mean_acc_m1 = np.mean(test_accuracies_m1)
            std_acc_m1 = np.std(test_accuracies_m1)

            accuracy_df = pd.DataFrame({'test_accuracies_m1':test_accuracies_m1,
                'mean_acc_m1':mean_acc_m1,
                'std_acc_m1':std_acc_m1})
            if save_path != None:
                accuracy_df.to_csv(f'{save_path}/classifier_acc_score.csv')

            print(f'Accuracy m1 : {mean_acc_m1}')


    ##############################################
    ########## BackBone / Rank Correlation #######
    ##############################################
    #Only BBBC dataset has a ground truth for continuity
    if params_preferences['dataset_tag']==1:
        # if 'GT_dist_toMax_phenotype' in MetaData_df.columns:
        #     MetaData_df = MetaData_df.rename(columns={'GT_dist_toMax_phenotype':'GT_dist_toInit_state'})
        #     to_GT = 'DataSets/MetaData1_GT_link_CP.csv'
        #     GT_df = pd.read_csv(to_GT,usecols=['Unique_ID','GT_initial_state'])
        #     MetaData_df = MetaData_df.join(GT_df.set_index('Unique_ID'), on='Unique_ID')
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
