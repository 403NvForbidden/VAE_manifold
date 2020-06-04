# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-06-03T10:42:40+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-06-04T22:59:37+10:00


'''
Access to R packages from python,
in order to use the CORANKING package
which implement the LCMC score for
dimension reduction technics
'''

# Choosing a CRAN Mirror

import coranking
from coranking.metrics import trustworthiness, continuity, LCMC

import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
#Use an inference DataLoader to go throughout ALL DATASET, and save each input image
#as a 64x64x3 vector
#Intput data must be in shape of  num_sample X 64x64x3

#Can pre-compute rank-matrix for high dimensional input data to save time, but
#need to be certain that the order match the embedded data !

def compute_coranking(metadata_csv, feature_size, save_matrix=True):
    '''
    Compute coranking matrix between input data and projected data, from the raw
    data and latent code saved in csv file
    '''
    MetaData_csv = pd.read_csv(metadata_csv)


    data_raw = MetaData_csv[['feature'+str(i) for i in range(feature_size)]].to_numpy()
    data_embedded = MetaData_csv[['x_coord','y_coord','z_coord']].to_numpy()


    Q_final = coranking.coranking_matrix(data_raw, data_embedded)

    if save_matrix:
        part_name = metadata_csv.split('_')
        name_pkl = f'DataSets/coranking_matrix_{part_name[-2]}_{part_name[-1]}.pkl'
        with open(name_pkl, 'wb') as f:
            pkl.dump(Q_final, f, protocol=pkl.HIGHEST_PROTOCOL)

    return Q_final


csv3 = 'DataSets/Sacha_Metadata_3dlatentVAEbigFAIL_20200525.csv'
_ = compute_coranking(csv3,64*64*3,save_matrix=True)


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

Q1_pkl = 'DataSets/coranking_matrix_3dlatentVAE_20200523.csv.pkl'
Q2_pkl = 'DataSets/coranking_matrix_3dlatentVAEFAIL_20200524.csv.pkl'
Q3_pkl = 'DataSets/coranking_matrix_3dlatentVAEbigFAIL_20200525.csv.pkl'
with open(Q1_pkl, 'rb') as f:
    Q1 = pkl.load(f)
with open(Q2_pkl, 'rb') as f:
    Q2 = pkl.load(f)
with open(Q3_pkl, 'rb') as f:
    Q3 = pkl.load(f)
print('All matrix are loaded')
trust1, cont1, lcmc1 = unsupervised_score(Q1)
print('Metrics for first model computed')
trust2, cont2, lcmc2 = unsupervised_score(Q2)
trust3, cont3, lcmc3 = unsupervised_score(Q3)
# RUN IN 16 minutes -- 5 min per model
#%%
fig, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True,figsize=(12,12))
ax1.plot(trust1,label='Model1')
ax1.plot(trust2,label='Model2')
#ax1.set_title('Trustworthiness')
#ax1.set_xticklabels([str(i)+'%' for i in range(1,20)])
ax3.set_xlabel('Neighborhood size - % of total datasize')
ax3.set_xticks(range(0,20))
ax3.set_xticklabels(range(1,21))
ax1.legend()
ax2.plot(cont1,label='Model1')
ax2.plot(cont2,label='Model2')
#ax2.plot(cont3,label='Model3')
ax2.set_title('Continuity')
ax2.legend()
ax3.plot(lcmc1,label='Model1')
ax3.plot(lcmc2,label='Model2')
#ax3.plot(lcmc3,label='Model3')
ax3.set_title('LCMC')
ax3.legend()
fig.suptitle('Unsupervised evaluation of projection at different scale')
plt.show()
