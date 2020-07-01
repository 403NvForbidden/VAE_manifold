# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-06-03T10:42:40+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-07-01T16:41:11+10:00


import coranking
from coranking.metrics import trustworthiness, continuity, LCMC
from local_quality import wt, ws
import torch
from torch import cuda
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.spatial import distance
from data_processing import get_inference_dataset
import itertools
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline


def unsup_metric_and_local_Q(metadata_csv,low_dim_names=['x_coord','y_coord','z_coord'],raw_data_included=False, feature_size=64*64*3, path_to_raw_data='DataSets/Synthetic_Data_1', saving_path=None, kt=300, ks=500):
    '''From a csv file containg 3D projection of single cell data, compute unsupervised metric
    (continuity, trustworthiness and LCMC), as well as return a local quality score per sample.

    :param raw_data_included: True if the raw data (high dim) is saved in the csv file under featurei columns for i from 0 to D-1
    '''

    ################################
    #### Process high and low dim data
    ################################

    print('##### Data is processed ...')
    MetaData_csv = pd.read_csv(metadata_csv)#,usecols=low_dim_names+['Unique_ID','GT_label'])
    data_embedded = None
    data_raw = None

    if raw_data_included: #high dim data is stored in csv
        data_embedded = MetaData_csv[low_dim_names].to_numpy()
        data_raw = MetaData_csv[['feature'+str(i) for i in range(feature_size)]].to_numpy()

    elif not(raw_data_included): #load image by batch, and save high dim data
        id_list = []
        list_of_tensors = [] #Store raw_data for performance metrics

        batch_size = 512
        input_size = 64
        _, dataloader = get_inference_dataset(path_to_raw_data,batch_size,input_size,droplast=False)

        for i, (data, labels, file_names) in enumerate(dataloader):
            #Extract unique cell id from file_names
            id_list.append([file_name for file_name in file_names])
            data = data.cuda()
            with torch.no_grad():
                raw_data = data.view(data.size(0),-1) #B x 64x64x3
                list_of_tensors.append(raw_data.data.cpu().numpy()) #Check if .data is necessary

            print(f'In progress...{i*len(data)}/{len(dataloader.dataset)}',end='\r')

        unique_ids = list(itertools.chain.from_iterable(id_list))
        raw_data = np.concatenate(list_of_tensors,axis=0)
        rawdata_frame = pd.DataFrame(data=raw_data[0:,0:],
                                index=[i for i in range(raw_data.shape[0])],
                               columns=['feature'+str(i) for i in range(raw_data.shape[1])])

        rawdata_frame['Unique_ID']=np.nan
        rawdata_frame.Unique_ID=unique_ids
        #rawdata_frame = rawdata_frame.sort_values(by=['Unique_ID'])
        #MetaData_csv = MetaData_csv.sort_values(by=['Unique_ID'])

        #Managed the fact that UMAP / tSNE might not contain all the single cells for some reason...
        true_size = len(rawdata_frame)
        rawdata_frame = rawdata_frame.join(MetaData_csv.set_index('Unique_ID'), on='Unique_ID')
        MetaData_csv = rawdata_frame
        MetaData_csv.dropna(subset=[low_dim_names[0]],inplace=True)
        new_size = len(MetaData_csv)
        print(f'{true_size-new_size} single cell were not find in the data projection!!!')

        data_embedded = MetaData_csv[low_dim_names].to_numpy()
        data_raw = MetaData_csv[['feature'+str(i) for i in range(feature_size)]].to_numpy()


    ##################################
    ## Compute coranking matrix and rank matrix
    ##################################

    print('##### Coranking Matrix computation ...')

    n, m = data_raw.shape
    #generated two n by n distance matrix. Data points stays in same order
    high_distance = distance.squareform(distance.pdist(data_raw))
    low_distance = distance.squareform(distance.pdist(data_embedded))

    #np.argsort return the indices that would sort an array
    # -. argsort two times to obtain the rank matrix
    high_ranking = high_distance.argsort(axis=1).argsort(axis=1)
    low_ranking = low_distance.argsort(axis=1).argsort(axis=1)
    #one line in high_ranking is a rho_ij for a fixed i and all j

    Q, xedges, yedges = np.histogram2d(high_ranking.flatten(),
                                       low_ranking.flatten(),
                                       bins=n)
    # Q = Coranking Matrix
    Q = Q[1:, 1:]  # remove rankings which correspond to themselves

    if saving_path != None:
        #part_name = metadata_csv.split('_')
        name_pkl = f'{saving_path}/{low_dim_names[0]}_coranking_matrix.pkl'
        with open(name_pkl, 'wb') as f:
            pkl.dump(Q, f, protocol=pkl.HIGHEST_PROTOCOL)

    #####################################
    ##### Compote local quality score (keep same order than csv)
    #####################################

    print('##### Local Q score computation ...')

    ws_mat = ws(high_ranking,low_ranking,ks)
    wt_mat = wt(high_ranking,low_ranking,kt)

    local_quality_score = 1./(2*ks*n) * (np.sum(ws_mat*wt_mat,axis=1) + (np.sum(ws_mat*wt_mat,axis=0)))
    # len = n , one score for each data point that correspond to the local quality score

    MetaData_csv['local_Q_score']=np.nan
    MetaData_csv.local_Q_score=local_quality_score

    light_df = MetaData_csv[low_dim_names+['Unique_ID','GT_label','local_Q_score']]
    if saving_path != None:
        light_df.to_csv(f'{saving_path}/{low_dim_names[0]}_light_metadata.csv')

    #####################################
    ##### Compute Unsupervised Score based on Q
    #####################################

    print('##### Unsupervised computation ...')

    trust, cont, lcmc = unsupervised_score(Q)

    aggregate_score = np.mean(np.stack([trust,cont,lcmc],axis=0),axis=0)
    x=range(20)
    trust_AUC = metrics.auc(x,trust)
    cont_AUC = metrics.auc(x,cont)
    lcmc_AUC = metrics.auc(x,lcmc)
    aggregate_AUC = metrics.auc(x,aggregate_score)

    if saving_path != None:
        unsup_score_df = pd.DataFrame({'trust':trust,'cont':cont,
            'lcmc':lcmc,'aggregate_score':aggregate_score,
            'trust_AUC':trust_AUC,'cont_AUC':cont_AUC,
            'lcmc_AUC':lcmc_AUC,'aggregate_AUC':aggregate_AUC})
        #Save the unsupervised_score to a CSV file
        unsup_score_df.to_csv(f'{saving_path}/{low_dim_names[0]}_unsupervised_score.csv')

    return trust_AUC, cont_AUC, lcmc_AUC, light_df


#Use an inference DataLoader to go throughout ALL DATASET, and save each input image
#as a 64x64x3 vector
#Intput data must be in shape of  num_sample X 64x64x3

#Can pre-compute rank-matrix for high dimensional input data to save time, but
#need to be certain that the order match the embedded data !

def compute_coranking(metadata_csv, feature_size, save_matrix=True, saving_path='DataSets/'):
    '''
    Compute coranking matrix between input data and projected data, from the raw
    data and latent code saved in csv file
    '''
    MetaData_csv = pd.read_csv(metadata_csv)


    data_raw = MetaData_csv[['feature'+str(i) for i in range(feature_size)]].to_numpy()
    data_embedded = MetaData_csv[['x_coord','y_coord','z_coord']].to_numpy()


    Q_final = coranking.coranking_matrix(data_raw, data_embedded)

    if save_matrix:
        #part_name = metadata_csv.split('_')
        name_pkl = f'{saving_path}coranking_matrix.pkl'
        with open(name_pkl, 'wb') as f:
            pkl.dump(Q_final, f, protocol=pkl.HIGHEST_PROTOCOL)

    return Q_final


#csv3 = 'DataSets/Sacha_Metadata_3dlatentVAEbigFAIL_20200525.csv'
#_ = compute_coranking(csv3,64*64*3,save_matrix=True)


def unsupervised_score(coranking_matrix):
    '''Compute different unsupervised performance metrics
    (trustworthiness, continuity and LCMC) for different neiborhood size,
    provided in list_of_k
    '''
    N = coranking_matrix.shape[0]
    #20 equally spaces neiborhood size between 1% and 20% of data size
    neighborhood_sizes = np.linspace(0.01*N,0.2*N,20).astype(np.int16)

    trust = trustworthiness(coranking_matrix, neighborhood_sizes)
    cont = continuity(coranking_matrix, neighborhood_sizes)
    lcmc = LCMC(coranking_matrix, neighborhood_sizes)

    return trust, cont, lcmc


def save_representation_plot(model_dataframe,saving_path,low_dim_names=['x_coord','y_coord','z_coord']):
    '''
    From a panda dataFrame that contains both latent code and local quality score
    of each data point (e.g, such a csv is saved after running 'unsup_metric_and_local_Q')
    save plotly 3D plot of learnt representation colored by :
    Plot 1) Ground truth cluster labels
    Plot 2) Local Quality score (show which region are more trustworthy)

    TO DO : if lmodel_dataframe is a LIST, a Local Quality Plot (Plot 2) is plotted for each
    model with the same colorbar scale, to be able to compare different model qualitatively,
    but no Plot 1.
    '''

    # GT cluster colored plot
    model_dataframe['GT_label'] = model_dataframe['GT_label'].astype(str)
    fig = px.scatter_3d(model_dataframe,x=low_dim_names[0],y=low_dim_names[1],z=low_dim_names[2],color='GT_label')
    plotly.offline.plot(fig, filename = saving_path+f'/{low_dim_names[0]}_GT_label.html', auto_open=False)

    # Local Q score colored plot
    fig = px.scatter_3d(light_df,x=low_dim_names[0],y=low_dim_names[1],z=low_dim_names[2],color='local_Q_score')
    plotly.offline.plot(fig, filename = save_path+f'/{low_dim_names[0]}_GT_local_score.html', auto_open=False)



# %% Put Both VAE and UMPA at same scale
VAE_csv = 'UMAP-tSNE/20200630_UMAP_vs_VAEx_coord_light_metadata.csv'
UMAP_csv = 'UMAP-tSNE/20200630_UMAP_vs_VAEUMAP_61_X_light_metadata.csv'

VAE_df = pd.read_csv(VAE_csv)
UMAP_df = pd.read_csv(UMAP_csv)
whithout_outliers = UMAP_df['UMAP_61_X'] > -100

min_color = np.min(VAE_df.local_Q_score.tolist()+UMAP_df[whithout_outliers].local_Q_score.tolist())
max_color = np.max(VAE_df.local_Q_score.tolist()+UMAP_df[whithout_outliers].local_Q_score.tolist())

low_dim_names = ['x_coord','y_coord','z_coord']
fig = px.scatter_3d(VAE_df,x=low_dim_names[0],y=low_dim_names[1],z=low_dim_names[2],color='local_Q_score',range_color=(min_color,max_color))
plotly.offline.plot(fig, filename = save_path+'/VAEa20b5_localscore_SCALED.html', auto_open=True)

low_dim_names = ['UMAP_61_X','UMAP_61_Y','UMAP_61_Z']
fig = px.scatter_3d(UMAP_df[whithout_outliers],x=low_dim_names[0],y=low_dim_names[1],z=low_dim_names[2],color='local_Q_score',range_color=(min_color,max_color))
plotly.offline.plot(fig, filename = save_path+'/UMAP61_localscore_SCALED.html', auto_open=True)



#%% Compute AUC
# x = range(20)
# trust_AUC = [metrics.auc(x,trust1),metrics.auc(x,trust2),metrics.auc(x,trust3)]
# cont_AUC = [metrics.auc(x,cont1),metrics.auc(x,cont2),metrics.auc(x,cont3)]
# lcmc_AUC = [metrics.auc(x,lcmc1),metrics.auc(x,lcmc2),metrics.auc(x,lcmc3)]
# #%%
# fig, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True,figsize=(12,12))
# ax1.plot(trust1,label=f'Model1 - AUC : {trust_AUC[0]:.2f}')
# ax1.plot(trust2,label=f'Model2 - AUC : {trust_AUC[1]:.2f}')
# #ax1.plot(trust3,label=f'Model3 - AUC : {trust_AUC[2]:.2f}')
# ax1.set_title('Trustworthiness')
# #ax1.set_xticklabels([str(i)+'%' for i in range(1,20)])
# ax3.set_xlabel('Neighborhood size - % of total datasize')
# ax3.set_xticks(range(0,20))
# ax3.set_xticklabels(range(1,21))
# ax1.legend()
# ax2.plot(cont1,label=f'Model1 - AUC : {cont_AUC[0]:.2f}')
# ax2.plot(cont2,label=f'Model2 - AUC : {cont_AUC[1]:.2f}')
# #ax2.plot(cont3,label=f'Model3 - AUC : {cont_AUC[2]:.2f}')
# ax2.set_title('Continuity')
# ax2.legend()
# ax3.plot(lcmc1,label=f'Model1 - AUC : {lcmc_AUC[0]:.2f}')
# ax3.plot(lcmc2,label=f'Model1 - AUC : {lcmc_AUC[1]:.2f}')
# #ax3.plot(lcmc3,label=f'Model1 - AUC : {lcmc_AUC[2]:.2f}')
# ax3.set_title('LCMC')
# ax3.legend()
# fig.suptitle('Unsupervised evaluation of projection at different scale')
# plt.show()
