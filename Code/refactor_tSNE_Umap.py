import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import plotly.offline
import torch
from torch.autograd import Variable
import umap

from quantitative_metrics.performance_metrics_single import compute_perf_metrics
from util.Process_benchmarkDataset import get_dsprites_inference_loader
from util.data_processing import get_inference_dataset
from util.helpers import metadata_latent_space, plot_from_csv
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

GT_csv_path = '/home/sachahai/Documents/VAE_manifold/DataSets/MetaData_FC_Felix_GT_link_CP.csv'
list_of_tensors, id_list = [], []


_, infer_dataloader = get_dsprites_inference_loader(batch_size=512, shuffle=True)
for i, (data, labels, file_names) in enumerate(infer_dataloader):
    # Extract unique cell id from file_names
    id_list.append([file_name for file_name in file_names])

    raw_data = data.view(data.size(0), -1)  # B x HxWxC
    list_of_tensors.append(raw_data)
###############
# %%

cols = ['feature' + str(i) for i in range(raw_data.shape[1])]
raw_data = np.concatenate(list_of_tensors, axis=0)
rawdata_frame = pd.DataFrame(data=raw_data[0:, 0:],
                             index=[i for i in range(raw_data.shape[0])],
                             columns=cols)
rawdata_frame['Unique_ID'] = np.nan
rawdata_frame.Unique_ID = list(itertools.chain.from_iterable(id_list))
df = rawdata_frame.sort_values(by=['Unique_ID'])
###############
# %% load data
###############
df = pd.read_csv('/home/sachahai/Documents/VAE_manifold/DataSets/Synthetic_Data_1/img_data.csv')

# %% load data


### run UMAP
embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      n_components=3,
                      metric='correlation').fit_transform(df.iloc[:, :-1].values)
umap_data = pd.DataFrame(embedding, columns=[f'umap_{n}' for n in range(3)])
umap_data['Unique_ID'] = df['Unique_ID']


MetaData_csv = pd.read_csv(GT_csv_path)


final = pd.merge(MetaData_csv, umap_data.set_index('Unique_ID'), how='outer', on=["Unique_ID"])
final.to_csv('/mnt/Linux_Storage/outputs/2_dsprite/UMAP/MetaDATA_umap.csv', index=False)

figplotly = plot_from_csv(final, low_dim_names=['umap_0', 'umap_1', 'umap_2'], GT_col='GT_class', dim=3,
                  column=None, as_str=True)
plotly.offline.plot(figplotly, filename=('/mnt/Linux_Storage/outputs/2_dsprite/UMAP/umap.html'), auto_open=True)

# %% load data
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(df.iloc[:, :-1].values)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

time_start = time.time()
tsne = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

tsne_data = pd.DataFrame(tsne_pca_results, columns=[f'tsne_{n}' for n in range(3)])
tsne_data['Unique_ID'] = df['Unique_ID']

final_tsne = pd.merge(MetaData_csv, tsne_data.set_index('Unique_ID'), how='outer', on=["Unique_ID"])
final_tsne.to_csv('/mnt/Linux_Storage/outputs/1_Felix/t-SNE/MetaDATA_umap.csv', index=False)
figplotly = plot_from_csv(final_tsne, low_dim_names=['tsne_0', 'tsne_1', 'tsne_2'], GT_col='GT_class', dim=3,
                  column=None, as_str=True)
plotly.offline.plot(figplotly, filename=('/mnt/Linux_Storage/outputs/1_experiment/t-SNE/tsne.html'), auto_open=True)
final_tsne.to_csv("/mnt/Linux_Storage/outputs/1_Felix/t-SNE/tsne.csv", index=False)