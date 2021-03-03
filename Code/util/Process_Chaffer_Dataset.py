# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-07-24T09:33:06+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-23T18:57:14+10:00

'''
Preprocessing off Dataset 3 -> Chaffer Dataset
Running this script merge all the channels of each single cell, save image as tiff
file in a folder architecture that suits the DataLoader used to train the VAEs.

Dataset is originally given as well plate images. Single cells images are extracted
from a standard Cell Profiler Pipeline.
CellProfiler output single_cell images (from segmentation) in a given folder architecture :
a separate folder for each channel
This code enable to regroupe images of same single cell in one HxWx3 tiff file

It is here that each cell is associated with a Unique ID, and the quantitative data from cell profiler
is matched with the groud truth data (Based on the centroid info of the cells)

A CSV File is saved, containing the matched info about each cell of Cell Profiler and Ground Truth
'''

import pandas as pd
import numpy as np
from skimage import io
import os
import matplotlib.pyplot as plt
import shutil

root = '/mnt/Linux_Storage/VAE-Manifold/Raw Datasets/Data_Chaffer/real_world_chaffer'
#single cell png files
single_cell_image_folder = os.path.join(root, 'per_channel_subset_sCells/')
#CSV file with ground truth (GT_label, Unique_ID and CP info)
GT_csv_read = os.path.join(root, '200809_Chaffer ground truth clusters_SUBSET_Only_Embedding_Calculations.csv')
GT_csv_save = 'DataSets/real_world_chaffer/MetaData3_Chaffer_GT_link_CP.csv'
#SAVE ONLY COLUMN OF INTEREST
GT_csv = pd.read_csv(GT_csv_read)
umap_df = GT_csv.filter(regex='^UMAP',axis=1).filter(regex='[XYZ]$',axis=1)
umap_cols = list(umap_df.columns)
tsne_df = GT_csv.filter(regex='^RTSNE',axis=1).filter(regex='[XYZ]$',axis=1)
tsne_cols = list(tsne_df.columns)
misc_cols = ['Manual_Clusters','Cell_Condition_status','Unique_Cell_ID','PCA_X','PCA_Y','PCA_Z','row ID','Label','Source Labeling','Well_categ','Cell_Type_categ','CD44_Level_categ','CD104_Level_categ','Treatment_categ','Designation_categ','Cell_Number_categ']



GT_csv = pd.read_csv(GT_csv_read,usecols=umap_cols+tsne_cols+misc_cols)

#Create DataFolder architecture
save_folder = 'DataSets/Chaffer_Data/'
list_folder = ['Class_'+str(i) for i in range(1,13)]
for folder in list_folder:
    saving_folder = f'{save_folder}{folder}'
    if os.path.exists(saving_folder):
        shutil.rmtree(saving_folder)
    os.makedirs(saving_folder)


### Add new needed column in GT csv file
# Unique_ID : cell file name   ,  GT_label : # link to the class
GT_csv['Unique_ID'] = np.nan
GT_csv['GT_label'] = np.nan

### Map class to a label number
switcher_class_to_label = {
    'HCC38HI_Not Specified_Not Specified_DHT_Not Specified': 1,
    'HCC38LO_Not Specified_Not Specified_DMSO_Not Specified': 2,
    'HMLER_Primed_Not Specified_DMSO_C2': 3,
    'HMLER_HIGH_Positive_DMSO_E3': 4,
    'HCC38LO_Not Specified_Not Specified_DHT_Not Specified': 5,
    'HMLER_Primed_Not Specified_DHT_C2': 6,
    'HMLER_HIGH_Positive_DHT_E3': 7,
    'HCC38HI_Not Specified_Not Specified_DMSO_Not Specified': 8,
    'HMLER_Primed_Not Specified_DMSO_C4': 9,
    'HMLER_HIGH_Positive_DMSO_F11': 10,
    'HMLER_Primed_Not Specified_DHT_C4': 11,
    'HMLER_HIGH_Positive_DHT_F11': 12
}

GT_csv['Unique_Cell_ID'] = GT_csv['Unique_Cell_ID'].astype(str)
GT_csv['Cell_Condition_status'] = GT_csv['Cell_Condition_status'].astype(str)

#list all the single cell file name (from folder of one channel)
all_single_cells = [f for f in os.listdir(single_cell_image_folder+'Phalloidin_images_mateched_to_subset_data/') if not f.startswith('.')]
counter =0
for cell_file_name in all_single_cells:

    file_wo_ext = cell_file_name.split('.')[0]
    file_name_save = file_wo_ext+'.tiff'

    img_c1 = io.imread(single_cell_image_folder+'Phalloidin_images_mateched_to_subset_data/'+cell_file_name)
    img_c2 = io.imread(single_cell_image_folder+'DAPI_images_mateched_to_subset_data/'+cell_file_name)
    img_c3 = io.imread(single_cell_image_folder+'AR_images_mateched_to_subset_data/'+cell_file_name)
    img_c4 = io.imread(single_cell_image_folder+'Zeb1_images_mateched_to_subset_data/'+cell_file_name)


    if not(img_c1.shape==img_c2.shape==img_c3.shape==img_c4.shape):
        counter+= 1
        continue

    final_img = np.stack([img_c1,img_c2,img_c3,img_c4],axis=-1)

    row = GT_csv['Unique_Cell_ID']==file_wo_ext
    idx = GT_csv.index[row]

    manual_class = GT_csv[row].Cell_Condition_status.item()
    class_num = switcher_class_to_label.get(manual_class)
    GT_csv.loc[idx,'GT_label'] = class_num
    GT_csv.loc[idx,'Unique_ID'] = file_name_save

    save_to = save_folder+f'Class_{class_num}/'+file_name_save
    io.imsave(save_to,final_img,plugin='tifffile')

### more or less 50 GT single cells are not present in the png files folder.
print(f'{counter} cells were NOT in the right shape format')
GT_csv.dropna(subset=['GT_label'],inplace=True)
GT_csv.to_csv(GT_csv_save,index=False)


#64 cells were not in the right format !!!

#%%
############# Statistics on the dataset 3 ###############
#########################################################
Metadata1 = pd.read_csv('DataSets/MetaData3_Chaffer_GT_link_CP.csv')

list_name=['HCC38LO-DMSO-CD44_Not Specified-CD104_Not Specified','HCC38LO-DHT-CD44_Not Specified-CD104_Not Specified','HMLER-DHT-CD44_Primed-CD104_Not Specified','HMLER-DMSO-CD44_Primed-CD104_Not Specified','HMLER-DHT-CD44_HIGH-CD104_Positive','HMLER-DMSO-CD44_HIGH-CD104_Positive','HCC38HI-DHT-CD44_Not Specified-CD104_Not Specified','HCC38HI-DMSO-CD44_Not Specified-CD104_Not Specified']
cluster_count = []
for i in range(8):
    value = Metadata1[Metadata1['Sub_population']==list_name[i]]['Unique_ID'].count()
    cluster_count.append(value)

cluster_count = np.array(cluster_count)
total_cell = np.sum(np.array(cluster_count))
cluster_count
cmap1 = plt.get_cmap('tab20')
colors1 = cmap1(np.linspace(0,1.0,20))
cmap2 = plt.get_cmap('Set3')
colors2 = cmap2(np.linspace(0,1.0,10))
colors = np.concatenate((colors1,colors2),0)
plt.figure(figsize=(14,6),dpi=300)
plt.bar(range(1,9),cluster_count,color=colors[[0,1,2,3,4,5,6,7]])
for i, pos in enumerate(cluster_count):
    plt.text(i+1,pos+30,str(pos)+' ('+str(np.round(pos/np.sum(cluster_count)*100,1))+'%)',ha='center',color='black',fontweight='bold')
plt.title(f'Dataset 3 cluster distribution ({len(Metadata1)} cells in total)')
# plt.bar(range(1,9),class_miss,color=colors[[0,1,2,3,4,5,6,7]])
# for i, pos in enumerate(class_miss):
#     plt.text(i+1,pos+2,str(pos),ha='center',color='black',fontweight='bold')
# plt.title(f'Single cells not detected by Cell Profiler ({np.sum(class_miss)} cells in total)')
plt.xticks(np.arange(1,9))
plt.ylabel('# of cells')
plt.xlabel('Cluster')
plt.savefig('dataset3_class_distribution.png')


#%%
# path_to_csv = 'DataSets/MetaData3_Chaffer_GT_link_CP.csv'
# df = pd.read_csv(path_to_csv)
# #df['Sub_population'] = df['Cell_Type_categ'].map(str) + '-' + df['Treatment_categ'].map(str) + '-CD44_' + df['CD44_Level_categ'].map(str) + '-CD104_' + df['CD104_Level_categ'].map(str)
# #df.head()
# #df.to_csv('DataSets/MetaData3_Chaffer_GT_link_CP.csv', index=False)
# cols = ['Unique_ID','VAE_x_coord','VAE_y_coord','VAE_z_coord']
# VAE_df = pd.read_csv('optimization/InfoMAX_VAE/Dataset3/run_Final_Chaffer_InfoM_alpha-beta_2020-08-10/models/alpha_20_beta_1/3chan_3z_InfoMAX_a_20_b_1_2020-08-11training_metadata.csv',usecols=cols)
# df = df.join(VAE_df.set_index('Unique_ID'),on='Unique_ID')
# df.head()
# #df.to_csv('DataSets/MetaData3_Chaffer_GT_link_CP.csv', index=False)
