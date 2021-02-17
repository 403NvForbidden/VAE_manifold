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
'''
import json
import warnings

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

from quantitative_metrics.disentanglement_metrics import compute_mig, compute_sap_score, \
    compute_interpretability_metric, compute_modularity, compute_correlation_score
from util.helpers import plot_from_csv, make_path

from quantitative_metrics.unsupervised_metric import unsup_metric_and_local_Q, save_representation_plot
from quantitative_metrics.MINE_metric import compute_MI
from quantitative_metrics.classifier_metric import classifier_performance
from quantitative_metrics.backbone_metric import dist_preservation_err


def compute_perf_metrics(data_source, params_preferences):
    '''
    From a csv or a pandas DataFrame containing the projection (latent code) of
    the dataset, compute and save different performance metrics, that can be used
    to evaluate the quality of the learnt representation.
    '''

    if isinstance(data_source, str):
        MetaData_df = pd.read_csv(data_source)
    else:
        MetaData_df = data_source

    ##############################################
    ########## Unsupervised Metrics ##############
    ##############################################
    try:
        if params_preferences['save_unsupervised_metric'] != False:
            save_path = os.path.join(params_preferences['global_saving_path'], "unsup_metric")
            make_path(save_path)

            print('#######################################')
            print('###### Computing unsupervised score')
            print('#######################################')

            trust_AUC, cont_AUC, lcmc_AUC, light_df = unsup_metric_and_local_Q(MetaData_df,
                                                                               low_dim_names=params_preferences[
                                                                                   'low_dim_names'],
                                                                               raw_data_included=True if len(
                                                                                   data_source.columns) > 10000 else False,
                                                                               feature_size=params_preferences[
                                                                                   'feature_size'],
                                                                               path_to_raw_data=params_preferences[
                                                                                   'path_to_raw_data'],
                                                                               saving_path=save_path,
                                                                               kt=params_preferences['kt'],
                                                                               ks=params_preferences['ks'],
                                                                               only_local_Q=params_preferences[
                                                                                   'only_local_Q'])

            # MetaData_df = light_df

            save_representation_plot(light_df, save_path, low_dim_names=params_preferences['low_dim_names'])

            print(f'Trustworthiness AUC : {trust_AUC}')
            print(f'Continuity AUC : {cont_AUC}')
            print(f'LCMC AUC : {lcmc_AUC}')
    except:
        warnings.warn("Unsupervised Metrics FAILED")
    ##############################################
    ########## Mutual Information ##############
    ##############################################
    try:
        if params_preferences['save_mine_metric'] != False:
            save_path = os.path.join(params_preferences['global_saving_path'], "MI_metric")
            make_path(save_path)

            print('#######################################')
            print('###### Computing Mutual Information')
            print('#######################################')

            MI_score = compute_MI(MetaData_df, low_dim_names=params_preferences['low_dim_names'],
                                  path_to_raw_data=params_preferences['path_to_raw_data'], save_path=save_path,
                                  batch_size=params_preferences['batch_size'],
                                  alpha_logit=params_preferences['alpha_logit'],
                                  bound_type=params_preferences['bound_type'], epochs=params_preferences['epochs'])

            MI_score_df = pd.DataFrame({'MI_score': MI_score}, index=[0])
            if save_path != '': MI_score_df.to_csv(f'{save_path}/MI_score.csv', index=False)

            print(f'Mutual Information : {MI_score}')
    except:
        warnings.warn("MI failed")
    ##############################################
    ########## Classifier accuracy ##############
    ##############################################
    try:
        if params_preferences['save_classifier_metric'] != False:
            save_path = os.path.join(params_preferences['global_saving_path'], "classifier_metric")
            make_path(save_path)

            print('#######################################')
            print('###### Computing Classifier accuracy')
            print('#######################################')

            # BBBC dataset has 3 different classifier metrics.
            if params_preferences['dataset_tag'] == 1:
                print('Metric 1...')
                # Metric 1 : Acc on all test single cells except uniform cluster 7
                _, test_accuracies_m1, _ = classifier_performance(MetaData_df,
                                                                  low_dim_names=params_preferences['low_dim_names'],
                                                                  Metrics=[True, False, False],
                                                                  num_iteration=params_preferences['num_iteration'])
                print('Metric 2...')
                # Metric 2 : Acc on all strong phenotypical change test single cells except uniform cluster 7
                _, test_accuracies_m2, _ = classifier_performance(MetaData_df,
                                                                  low_dim_names=params_preferences['low_dim_names'],
                                                                  Metrics=[False, True, False],
                                                                  num_iteration=params_preferences['num_iteration'])
                print('Metric 3...')
                # Metric 3 : Acc on all strong phenotypical change + META_CLUSTER (1&2, 3&4 and 5&6 grouped) test single cells except uniform cluster 7
                _, test_accuracies_m3, _ = classifier_performance(MetaData_df,
                                                                  low_dim_names=params_preferences['low_dim_names'],
                                                                  Metrics=[False, False, True],
                                                                  num_iteration=params_preferences['num_iteration'])

                mean_acc_m1 = np.mean(test_accuracies_m1)
                std_acc_m1 = np.std(test_accuracies_m1)
                mean_acc_m2 = np.mean(test_accuracies_m2)
                std_acc_m2 = np.std(test_accuracies_m2)
                mean_acc_m3 = np.mean(test_accuracies_m3)
                std_acc_m3 = np.std(test_accuracies_m3)

                accuracy_df = pd.DataFrame(
                    {'test_accuracies_m1': test_accuracies_m1, 'test_accuracies_m2': test_accuracies_m2,
                     'test_accuracies_m3': test_accuracies_m3, 'mean_acc_m1': mean_acc_m1,
                     'std_acc_m1': std_acc_m1, 'mean_acc_m2': mean_acc_m2,
                     'std_acc_m2': std_acc_m2, 'mean_acc_m3': mean_acc_m3, 'std_acc_m3': std_acc_m3})
                if save_path != '': accuracy_df.to_csv(f'{save_path}/classifier_acc_score.csv', index=False)

                print(f'Accuracy m1 : {mean_acc_m1}')
                print(f'Accuracy m2 : {mean_acc_m2}')
                print(f'Accuracy m3 : {mean_acc_m3}')

            else:

                print('Metric 1...')
                if params_preferences['dataset_tag'] == 2:  # Horvath dataset, imbalanced and 6 class
                    _, test_accuracies_m1, _ = classifier_performance(MetaData_df,
                                                                      low_dim_names=params_preferences['low_dim_names'],
                                                                      Metrics=[True, False, False],
                                                                      num_iteration=params_preferences['num_iteration'],
                                                                      num_class=6, class_to_ignore='None',
                                                                      imbalanced_data=True)
                elif params_preferences['dataset_tag'] == 3:  # Chaffer dataset,
                    _, test_accuracies_m1, _ = classifier_performance(MetaData_df,
                                                                      low_dim_names=params_preferences['low_dim_names'],
                                                                      Metrics=[True, False, False],
                                                                      num_iteration=params_preferences['num_iteration'],
                                                                      num_class=12, class_to_ignore='None',
                                                                      imbalanced_data=True)

                mean_acc_m1 = np.mean(test_accuracies_m1)
                std_acc_m1 = np.std(test_accuracies_m1)

                accuracy_df = pd.DataFrame({'test_accuracies_m1': test_accuracies_m1,
                                            'mean_acc_m1': mean_acc_m1,
                                            'std_acc_m1': std_acc_m1})
                if save_path != '': accuracy_df.to_csv(f'{save_path}/classifier_acc_score.csv', index=False)

                print(f'Accuracy m1 : {mean_acc_m1}')
    except:
        warnings.warn("Classifier metrics FAILED")
    ##############################################
    ########## BackBone / Rank Correlation #######
    ##############################################
    # Only BBBC dataset has a ground truth for continuity
    try:
        if params_preferences['dataset_tag'] == 1 and params_preferences['save_backbone_metric']:
            save_path = os.path.join(params_preferences['global_saving_path'], "backbone_metric")
            make_path(save_path)

            print('#######################################')
            print('###### Computing Backbone Metric')
            print('#######################################')

            _, spearman_r, kendall_r = dist_preservation_err(MetaData_df,
                                                             low_dim_names=params_preferences['low_dim_names'],
                                                             overwrite_csv=False, save_path=save_path)

            print(f'Spearman Coefficient : {spearman_r}')
            print(f'Kendall Coefficient : {kendall_r}')
    except:
        warnings.warn("BackBone FAILED")
    ##############################################
    ########## Disentanglement Metrics  ##########
    ##############################################
    try:
        # Only BBBC dataset and sprite dataset has a ground truth for disentanglement
        if params_preferences['save_disentanglement_metric'] and params_preferences['dataset_tag'] == 1:
            save_path = os.path.join(params_preferences['global_saving_path'], "disentanglement_metric")
            make_path(save_path)

            features = data_source[params_preferences['features']].values
            latent = data_source[params_preferences['low_dim_names']].values
            res = {
                'mig': compute_mig(latent, features),
                'sap': compute_sap_score(latent, features),
                'interpretability': compute_interpretability_metric(latent, features, params_preferences['features']),
                'modularity': compute_modularity(latent, features),
                'corr': compute_correlation_score(latent, features)
            }
            print(res)  # TODO: write a func to plot it
            if save_path != '':
                with open(f'{save_path}/disentangelement_score.json', 'w') as fp:
                    json.dump(res, fp)
    except:
        warnings.warn("Disentanglement Metrics FAILED")

    plt.close('all')
    print('Metrics Computation Terminated')
