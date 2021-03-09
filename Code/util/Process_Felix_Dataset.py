# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-06-07T21:34:58+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-23T19:02:47+10:00

'''
Preprocessing off Dataset 2 -> Horvath Dataset
This file contains all the steps necessary to process the Peter Horvath
synthetic dataset, in a way it could be fed to the VAE framework
Namely it will link all the single cell output from CellProfiler pipeline to
the ground truth from Peter Horvath that contains cell location and class label

From the ground truth position, associate the closest centroid detected by CellProfiler
(if in a given region, otherwise the cell is disregarded)

It is here that each cell is associated with a Unique ID, and the quantitative data from cell profiler
is matched with the groud truth data (Based on the centroid info of the cells)

A CSV File is saved, containing the matched info about each cell of Cell Profiler and Ground Truth

'''

import os
import shutil
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from glob import glob
import warnings

from skimage import io

id = "Zeb1" # "Ecad"
### Path to CellProfiler Outputs
root = '/mnt/Linux_Storage/VAE-Manifold/Raw Datasets/Felix_honours_data'

path_to_CP = os.path.join(root, f'200405 F1 SULF1_{id.upper()} n1/DAPI_Phall_Sulf1_{id}/per_cell_cropped_images')#'200405 F1 SULF1_ZEB1 n1/DAPI_Phall_Sulf1_Zeb1/per_cell_cropped_images/')

GT_csv = os.path.join(root, f'200405 F1 SULF1_{id.upper()} n1/DAPI_Phall_Sulf1_{id}/Quantitative_Data/200404_DAPI_Phall_SULF1_{id.upper()}_Quantitative_DataCell_Bodies.csv')# '200405 F1 SULF1_ZEB1 n1/DAPI_Phall_Sulf1_Zeb1/Quantitative_Data/200404_DAPI_Phall_SULF1_ZEB1_Quantitative_DataCell_Bodies.csv')

# Extraction of useful information from CellProfiler MetaData
fields = ['ImageNumber','ObjectNumber','Metadata_Coverslip','Neighbors_PercentTouching_Adjacent','AreaShape_Center_X','AreaShape_Center_Y'] # 'Metadata_Channel'
CP_df = pd.read_csv(GT_csv, usecols=fields)

path_to_save = "/home/sachahai/Documents/VAE_manifold/DataSets"

# %%
### Path to SAVING folder. Will contains 8 subfolders corresponding to the 8
# discrete class of the dataset.


save_folder = os.path.join(path_to_save, 'Felix_channelwise')

list_folder = ['Class_'+str(i) for i in range(1,5)] #Will consider ONLY Cluster 1 to 6
'''
for folder in list_folder:
    saving_folder = os.path.join(save_folder, folder)
    if os.path.exists(saving_folder):
        shutil.rmtree(saving_folder)
    os.makedirs(saving_folder)
'''

### Add new needed column in GT csv file
# Unique_ID : cell file name   ,  GT_label : # link to the class
CP_df['Unique_ID'] = np.nan
CP_df['GT_label'] = np.nan
for c in range(1, 5):
    for index, row in CP_df.iterrows():
        #if row.Neighbors_PercentTouching_Adjacent == 100:
        #print(row.ImageNumber)
        img_c1_path = os.path.join(path_to_CP, f"200405_Coverslip{int(row.Metadata_Coverslip)}_XY{int(row.ImageNumber):02d}_Channel{c}/Cell_Bodies_{int(row.ObjectNumber)}.png")
        img = io.imread(img_c1_path)
        # img_c1 = io.imread(img_c1_path)
        # img_c2 = io.imread(img_c1_path.replace('Channel1', 'Channel2'))
        # img_c3 = io.imread(img_c1_path.replace('Channel1', 'Channel3'))
        # img_c4 = io.imread(img_c1_path.replace('Channel1', 'Channel4'))
        #except:
         #   continue
        
        #if not(img_c1.shape==img_c2.shape==img_c3.shape==img_c4.shape):
        #    warnings.warn(f"channel size not matching: {row}")
        #    continue    
        # final_img = np.stack([img_c1,img_c2,img_c3,img_c4],axis=-1)
        file_name_save = f'Coverslip{int(row.Metadata_Coverslip)}_XY{int(row.ImageNumber)}_{int(row.ObjectNumber)}' + '.tiff'
        CP_df['Unique_ID'][index] = file_name_save
        CP_df['GT_label'][index]  = row.Neighbors_PercentTouching_Adjacent
        save_to = os.path.join(save_folder, f'Class_{c}', file_name_save)
        print(img_c1_path, "----", save_to)
        #io.imsave(save_to, img, plugin='tifffile')
'''
CP_df = CP_df.drop(['ImageNumber', 'ObjectNumber', 'Metadata_Coverslip', 'Neighbors_PercentTouching_Adjacent'], axis = 1) 
CP_df.to_csv(os.path.join(save_folder, 'MetaData_FC_ZEB1_Felix_GT_link_CP.csv'), index=False)

CP_ecad_df = pd.read_csv(os.path.join(save_folder, 'MetaData_FC_Ecad_Felix_GT_link_CP.csv'))
CP_ecad_df['GT_dataset'] = 'Ecad'
CP_df['GT_dataset'] = 'ZEB1'

CP_new = pd.concat([CP_ecad_df, CP_df])
CP_new.to_csv(os.path.join(save_folder, 'MetaData_FC_Felix_GT_link_CP.csv'), index=False)
'''