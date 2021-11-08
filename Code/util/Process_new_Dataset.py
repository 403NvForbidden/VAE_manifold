import os
import shutil
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from glob import glob
import warnings

from skimage import io


### Path to CellProfiler Outputs
root = '/home/sachahai/Documents/VAE_manifold/DataSets/'
# %%
path_to_CP = os.path.join(root, f'4Channel_TIFs')#'200405 F1 SULF1_ZEB1 n1/DAPI_Phall_Sulf1_Zeb1/per_cell_cropped_images/')

GT_csv = os.path.join(root, f'210728_GZIF12_FIXED_dataset_parsed_embedding_categ_cell_ID.csv')# '200405 F1 SULF1_ZEB1 n1/DAPI_Phall_Sulf1_Zeb1/Quantitative_Data/200404_DAPI_Phall_SULF1_ZEB1_Quantitative_DataCell_Bodies.csv')

# Extraction of useful information from CellProfiler MetaData
fields = ['row ID','Unique_Cell_ID','Cell_Class_Prediction_categ']
CP_df = pd.read_csv(GT_csv, usecols=fields)

path_to_save = "/home/sachahai/Documents/VAE_manifold/DataSets"

# %%
### Path to SAVING folder. Will contains 8 subfolders corresponding to the 8
# discrete class of the dataset.
# went through the folder find the files with corresponding ground truth

save_folder = os.path.join(path_to_save, '4Channel_TIFs')
uniq_val, encoded_class = np.unique(CP_df.Cell_Class_Prediction_categ, return_inverse=True)
CP_df['GT_label'] = encoded_class
# %% create class directory
list_folder = ['Class_'+str(i) for i in range(len(uniq_val))] #Will consider ONLY Cluster 1 to 6
for folder in list_folder:
    saving_folder = os.path.join(save_folder, folder)
    if os.path.exists(saving_folder):
        shutil.rmtree(saving_folder)
    os.makedirs(saving_folder) 
# %%
CP_df['Unique_ID'] = np.nan
list_well_images = os.listdir(path_to_CP)
for img in list_well_images:
    if os.path.isdir(os.path.join(path_to_CP, img)): continue
    # find the corresponding GT label
    row = CP_df[CP_df['Unique_Cell_ID'] == img.strip(".tif")]
    # savle to coresponding folder 
    CP_df.loc[row.index].Unique_ID = img
    shutil.move(os.path.join(path_to_CP, img), os.path.join(path_to_CP, f"Class_{row.GT_label.values[0]}", img))
'''
for folder in list_folder: 
    list_imgs = os.listdir(os.path.join(path_to_CP, folder))
    for img in list_imgs:
        row = CP_df[CP_df['Unique_Cell_ID'] == img.strip(".tif")]
        CP_df.Unique_ID.loc[row.index] = img.strip(".tif")
    '''
# %%
df = pd.read_csv('/home/sachahai/Documents/VAE_manifold/DataSets/MetaData_NEW_GT_link_CP.csv')
df = df.rename(columns={'Unique_Cell_ID': 'Unique_ID'})
df = df.drop(['row ID'], axis=1)
# %%
# drop uniq id
CP_df = CP_df.drop(['Unique_Cell_ID'], axis=1)
CP_df.to_csv(os.path.join(save_folder, 'MetaData_NEW_GT_link_CP.csv'), index=False)