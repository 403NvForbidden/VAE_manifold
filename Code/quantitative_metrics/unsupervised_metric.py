# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-06-03T10:42:40+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-31T11:18:21+10:00

"""
Main file that contains all the function necessary to compute the unsupervised performance metrics
used to assess the quality of a learnt representation

Co-ranking matrix between high dimensional data and its projection is computed.
From it Trustworthiness, Continuity and LCMC metrics are extracted.

The contribution of each single point can also be computed to obtain a local quality score.
"""

import coranking
from coranking.metrics import trustworthiness, continuity, LCMC
from quantitative_metrics.local_quality import wt, ws
import torch
from torch import cuda
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import metrics
from scipy.spatial import distance
from util.data_processing import get_inference_dataset
import itertools
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline
device = torch.device('cpu' if not cuda.is_available() else 'cuda')

def unsup_metric_and_local_Q(metadata_csv, low_dim_names=['x_coord', 'y_coord', 'z_coord'], raw_data_included=False,
                             feature_size=64 * 64 * 3, path_to_raw_data='DataSets/Synthetic_Data_1', saving_path=None,
                             kt=300, ks=500, only_local_Q=False, logger=None):
    """
    From a csv file containg low dimensional projection of single cell data, compute unsupervised metric
    (continuity, trustworthiness and LCMC), as well as return a local quality score per sample.

    :parameter
        :param metadata_csv (string or DataFrame) : Path to csv file or directly DataFrame that contains the latent codes as well as ground information
        :param low_dim_names ([string]) : Names of the columns that stores the latent codes
        :param raw_data_included (bolean) : True if the raw data (high dim) is saved in the csv file under featurei columns for i from 0 to D-1
        :param feature_size (int) : Number of dimension in the high dimensional data
        :param path_to_raw_data (string) : Path to the folder where the raw data (single cell images) are stored
        :param saving_path (string) : Path to the folder where to store the results
        :param kt, ks (int) : Neighborhood size parameter for the local quality score (please refer to local_quality.py)
        :param only_local_Q (bolean) : If True, only the local quality score is computed (save computational time)
    """
    ################################
    #### Process high and low dim data
    ################################

    print('##### Data is processed ...')
    if isinstance(metadata_csv, str):
        MetaData_csv = pd.read_csv(metadata_csv)
    else:
        MetaData_csv = metadata_csv
    data_embedded = None
    data_raw = None

    if raw_data_included:  # high dim data is stored in csv
        data_embedded = MetaData_csv[low_dim_names].to_numpy()
        data_raw = MetaData_csv[['feature' + str(i) for i in range(feature_size)]].to_numpy()

    elif not (raw_data_included):  # load image by batch, and save high dim data
        id_list = []
        list_of_tensors = []  # Store raw_data for performance metrics

        batch_size = 512
        input_size = 64  # TO CHANGE DEPENDING ON THE DATASET ################
        _, dataloader = get_inference_dataset(path_to_raw_data, batch_size, input_size, droplast=False)

        for i, (data, labels, file_names) in enumerate(dataloader):
            # Extract unique cell id from file_names
            id_list.append([file_name for file_name in file_names])
            data = data.to(device)
            with torch.no_grad():
                raw_data = data.view(data.size(0), -1)  # B x 64x64x3
                list_of_tensors.append(raw_data.data.cpu().numpy())

            print(f'In progress...{i * len(data)}/{len(dataloader.dataset)}', end='\r')

        unique_ids = list(itertools.chain.from_iterable(id_list))
        raw_data = np.concatenate(list_of_tensors, axis=0)
        rawdata_frame = pd.DataFrame(data=raw_data[0:, 0:],
                                     index=[i for i in range(raw_data.shape[0])],
                                     columns=['feature' + str(i) for i in range(raw_data.shape[1])])

        rawdata_frame['Unique_ID'] = np.nan
        rawdata_frame.Unique_ID = unique_ids
        # rawdata_frame = rawdata_frame.sort_values(by=['Unique_ID'])
        # MetaData_csv = MetaData_csv.sort_values(by=['Unique_ID'])

        # Managed the fact that UMAP / tSNE might not contain all the single cells for some reason...
        true_size = len(rawdata_frame)
        rawdata_frame = rawdata_frame.join(MetaData_csv.set_index('Unique_ID'), on='Unique_ID')
        MetaData_csv = rawdata_frame
        MetaData_csv.dropna(subset=[low_dim_names[0]], inplace=True)
        new_size = len(MetaData_csv)
        print(f'{true_size - new_size} single cell were not find in the data projection!!!')

        data_embedded = MetaData_csv[low_dim_names].to_numpy()
        data_raw = MetaData_csv[['feature' + str(i) for i in range(feature_size)]].to_numpy()

    ##################################
    ## Compute coranking matrix and rank matrix
    ##################################

    print('##### Coranking Matrix computation ...')

    n, m = data_raw.shape
    # generated two n by n distance matrix. Data points stays in same order
    high_distance = distance.squareform(distance.pdist(data_raw))
    low_distance = distance.squareform(distance.pdist(data_embedded))

    # np.argsort return the indices that would sort an array
    # -. argsort two times to obtain the rank matrix
    high_ranking = high_distance.argsort(axis=1).argsort(axis=1)
    low_ranking = low_distance.argsort(axis=1).argsort(axis=1)
    # one line in high_ranking is a rho_ij for a fixed i and all j

    Q, xedges, yedges = np.histogram2d(high_ranking.flatten(),
                                       low_ranking.flatten(),
                                       bins=n)
    # Q = Coranking Matrix
    Q = Q[1:, 1:]  # remove rankings which correspond to themselves

    if saving_path != None:
        # part_name = metadata_csv.split('_')
        coranking_matrix_plot(Q, saving_path=saving_path, name=low_dim_names[0])
        name_pkl = f'{saving_path}/{low_dim_names[0]}_coranking_matrix.pkl'
        with open(name_pkl, 'wb') as f:
            pkl.dump(Q, f, protocol=pkl.HIGHEST_PROTOCOL)

    #####################################
    ##### Compote local quality score (keep same order than csv)
    #####################################

    print('##### Local Q score computation ...')

    ws_mat = ws(high_ranking, low_ranking, ks)
    wt_mat = wt(high_ranking, low_ranking, kt)

    local_quality_score = 1. / (2 * ks * n) * (np.sum(ws_mat * wt_mat, axis=1) + (np.sum(ws_mat * wt_mat, axis=0)))
    # len = n , one score for each data point that correspond to the local quality score

    MetaData_csv['local_Q_score'] = np.nan
    MetaData_csv.local_Q_score = local_quality_score

    # If old run of UMAP, some GT info are missing, manage that
    ## TODO:  to remove when all UMAP runs will be up to date
    if 'GT_dist_toMax_phenotype' in MetaData_csv.columns:
        MetaData_csv = MetaData_csv.rename(columns={'GT_dist_toMax_phenotype': 'GT_dist_toInit_state'})
        to_GT = 'DataSets/MetaData1_GT_link_CP.csv'
        GT_df = pd.read_csv(to_GT, usecols=['Unique_ID', 'GT_initial_state'])
        MetaData_csv = MetaData_csv.join(GT_df.set_index('Unique_ID'), on='Unique_ID')

    light_df = MetaData_csv[low_dim_names + ['Unique_ID', 'GT_label',
                                             'local_Q_score']]  # ,'GT_Shape','GT_dist_toInit_state','GT_initial_state']]
    if saving_path != None:
        light_df.to_csv(f'{saving_path}/{low_dim_names[0]}_light_metadata.csv')

    if only_local_Q:
        return None, None, None, light_df

    #####################################
    ##### Compute Unsupervised Score based on Q
    #####################################
    print('##### Unsupervised score computation ...')
    trust, cont, lcmc = unsupervised_score(Q)

    aggregate_score = np.mean(np.stack([trust, cont, lcmc], axis=0), axis=0)
    x = range(20)
    trust_AUC = metrics.auc(x, trust)
    cont_AUC = metrics.auc(x, cont)
    lcmc_AUC = metrics.auc(x, lcmc)
    aggregate_AUC = metrics.auc(x, aggregate_score)

    if saving_path != None:
        unsup_score_df = pd.DataFrame({'trust': trust, 'cont': cont,
                                       'lcmc': lcmc, 'aggregate_score': aggregate_score,
                                       'trust_AUC': trust_AUC, 'cont_AUC': cont_AUC,
                                       'lcmc_AUC': lcmc_AUC, 'aggregate_AUC': aggregate_AUC})
        # Save the unsupervised_score to a CSV file
        unsup_score_df.to_csv(f'{saving_path}/{low_dim_names[0]}_unsupervised_score.csv')
        fig = lcmc_curves_plot(trust, trust_AUC, cont, cont_AUC, lcmc, lcmc_AUC, saving_path=saving_path + f'/{low_dim_names[0]}_')
        if logger:
            logger.experiment.add_figure(tag="Unsuperivsed metriecs", figure=fig, close=True)

    return trust_AUC, cont_AUC, lcmc_AUC, light_df

def unsupervised_score(coranking_matrix):
    """Compute different unsupervised performance metrics
    (trustworthiness, continuity and LCMC) for different neiborhood size,
    provided in list_of_k
    """
    N = coranking_matrix.shape[0]
    # 20 equally spaces neiborhood size between 1% and 20% of data size
    neighborhood_sizes = np.linspace(0.01 * N, 0.2 * N, 20).astype(np.int16)

    trust = trustworthiness(coranking_matrix, neighborhood_sizes)
    cont = continuity(coranking_matrix, neighborhood_sizes)
    lcmc = LCMC(coranking_matrix, neighborhood_sizes)

    return trust, cont, lcmc


def coranking_matrix_plot(Q, saving_path=None, name=None):
    # Save CoRanking plot Image
    plt.figure(figsize=(6, 6))
    plt.imshow(Q[:800, :800], cmap=plt.cm.gnuplot2_r, norm=LogNorm())
    plt.title('First 800 Ranks - Coranking Matrix')
    plt.savefig(saving_path + f'/{name}_coranking_plot_zoom.png')
    plt.close()
    plt.figure(figsize=(6, 6))
    plt.imshow(Q, cmap=plt.cm.gnuplot2_r, norm=LogNorm())
    plt.title('Full Coranking Matrix')
    if saving_path != None:
        plt.savefig(saving_path + f'/{name}_coranking_plot.png')
    plt.close()


def lcmc_curves_plot(trust, trust_AUC, cont, cont_AUC, lcmc, lcmc_AUC, saving_path=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 12))
    ax1.plot(trust, label=f'Model - AUC : {trust_AUC:.2f}')
    ax1.set_title('Trustworthiness')
    ax1.legend()
    ax3.set_xlabel('Neighborhood size - % of total datasize')
    ax3.set_xticks(range(0, 20))
    ax3.set_xticklabels(range(1, 21))
    ax2.plot(cont, label=f'Model - AUC : {cont_AUC:.2f}')
    ax2.set_title('Continuity')
    ax2.legend()
    ax3.plot(lcmc, label=f'Model - AUC : {lcmc_AUC:.2f}')
    ax3.set_title('LCMC')
    ax3.legend()
    fig.suptitle('Unsupervised evaluation of projection at different scale')
    if saving_path != None:
        plt.savefig(saving_path + 'unsup_score_plot.png')
    plt.close()
    return fig


def save_representation_plot(model_dataframe, saving_path, low_dim_names=['x_coord', 'y_coord', 'z_coord']):
    """
    From a panda dataFrame that contains both latent code and local quality score
    of each data point (e.g, such a csv is saved after running 'unsup_metric_and_local_Q')
    save plotly 3D plot of learnt representation colored by :
    Plot 1) Ground truth cluster labels
    Plot 2) Local Quality score (show which region are more trustworthy)

    TO DO : if lmodel_dataframe is a LIST, a Local Quality Plot (Plot 2) is plotted for each
    model with the same colorbar scale, to be able to compare different model qualitatively,
    but no Plot 1.
    """

    # GT cluster colored plot
    model_dataframe['GT_label'] = model_dataframe['GT_label'].astype(str)
    fig = px.scatter_3d(model_dataframe, x=low_dim_names[0], y=low_dim_names[1], z=low_dim_names[2], color='GT_label')
    if saving_path != None:
        plotly.offline.plot(fig, filename=saving_path + f'/{low_dim_names[0]}_GT_label.html', auto_open=False)

    # Local Q score colored plot
    fig = px.scatter_3d(model_dataframe, x=low_dim_names[0], y=low_dim_names[1], z=low_dim_names[2],
                        color='local_Q_score')
    if saving_path != None:
        plotly.offline.plot(fig, filename=saving_path + f'/{low_dim_names[0]}_GT_local_score.html', auto_open=False)
