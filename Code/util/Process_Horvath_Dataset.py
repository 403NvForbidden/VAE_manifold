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

from skimage import io

### Path to CellProfiler Outputs
root = '/mnt/Linux_Storage/VAE-Manifold/Raw Datasets/Data_Horvath'
path_to_CP = os.path.join(root, 'Horvath_Synth_Complex_Dataset/CellProfiler_Outputs/')

CP_Quantitative_CSV = os.path.join(path_to_CP, 'Quantitative_Outputs/200602_Horvath_Complex_AnalysisSplitCellBodies.csv')

# Extraction of useful information from CellProfiler MetaData
fields = ['ImageNumber','ObjectNumber','Metadata_Plate','Metadata_Well','AreaShape_Center_X','AreaShape_Center_Y']
#Metadata_Plate is the synthetic plate number (1-32)
#Metadata_Well is the Well (letter-number)

CP_df = pd.read_csv(CP_Quantitative_CSV,usecols=fields)

# %% Control that info match with an example image
img_test = os.path.join(root, 'db11 - synthetic_peter_2 - Copy/synthPlate001/anal1/synthPlate001_wA02.png')
img = io.imread(img_test)
fig, ax = plt.subplots(1,1,figsize=(10, 10))

site1 = CP_df['Metadata_Plate']==1
Well1 = CP_df['Metadata_Well']=='A02'

x = CP_df[site1 & Well1]['AreaShape_Center_X']
y = CP_df[site1 & Well1]['AreaShape_Center_Y']
ax.scatter(x.values,y.values,s=60,c='red',marker='X')
ax.imshow(img)

# Everything look ok !!

### Path to Peter Horvath Ground truth File
GT_path = '../Data_Horvath/db11 - synthetic_peter_2 - Copy/'

def closest_point(point,points):
    points = np.asarray(points)
    dist_2 = np.sum((points-point)**2, axis=1)
    return np.argmin(dist_2), dist_2[np.argmin(dist_2)]

# %%
### Path to SAVING folder. Will contains 8 subfolders corresponding to the 8
# discrete class of the dataset.

save_folder = 'DataSets/Peter_Horvath_Data/'
list_folder = ['Class_'+str(i) for i in range(1,7)] #Will consider ONLY Cluster 1 to 6
for folder in list_folder:
    saving_folder = f'{save_folder}{folder}'
    if os.path.exists(saving_folder):
        shutil.rmtree(saving_folder)
    os.makedirs(saving_folder)

#Future CSV file in which all metadata will be stored
MetaData_GT_link_CP = pd.DataFrame(columns=['Plate','Well','GT_label','Unique_ID','GT_x','GT_y','CP_ImagerNumber','CP_ObjectNumber','CP_x','CP_y'])

# %%
all_plates_folder = [f for f in os.listdir(GT_path) if not f.startswith('.')]
all_plates_folder.sort()

#How many subfolder to process (Choose between 1 to 32) 32 lead to full datasize (300'000 cells)
n_plates = 32

missed_cells = 0
class_miss = np.zeros(8)
for plate in all_plates_folder:

    path_to_plate = plate+'/anal2/'
    synth_plate = plate[-3:]
    print(f'--------Considering Plate {synth_plate} -----------------')

    #For a given synthetic plate, there is one .class ground truth file per well image
    #Loop over each class file
    list_well_images = os.listdir(GT_path+path_to_plate)

    for well_image in list_well_images:
        if (well_image.endswith('.class') or well_image.endswith('.csv')):

            #GT comes weirdly in non-formated .class file. Simply rename them and treat them as csv
            if (well_image.endswith('.class')):
                pre, ext = os.path.splitext(GT_path+path_to_plate+well_image)
                os.rename(GT_path+path_to_plate+well_image, pre+'.csv')

            pre, ext = os.path.splitext(GT_path+path_to_plate+well_image)

            #Consider only CP metada for the given plate and well
            plate_i = synth_plate.lstrip('0')
            no_ext = well_image.split('.')
            well_name = no_ext[0].split('_')
            well_i = well_name[-1][1:]

            Platei_CP = CP_df['Metadata_Plate']==int(plate_i)
            Welli_CP = CP_df['Metadata_Well']==well_i
            CP_Pi_Wi = CP_df[Platei_CP & Welli_CP].reset_index()
            CP_centroids_array = CP_Pi_Wi[['AreaShape_Center_X','AreaShape_Center_Y']].values


            #GT dataframe associate with this plate - well
            GT_df = pd.read_csv(pre+'.csv',header=None, names=['GT_pos_y', 'GT_pos_x', 'GT_label'])

            #iterate over each SINGLE CELL (row of the GT file of a given well image)
            for index, row in GT_df.iterrows():
                GT_centroid = np.array([row['GT_pos_x'],row['GT_pos_y']])

                #Find the closest cell profiler centroid
                if (len(CP_centroids_array) > 0):
                    CP_row_min, dist_2 = closest_point(GT_centroid,CP_centroids_array)
                    CP_centroid = CP_centroids_array[CP_row_min]
                else:
                    print(f'Cell Profiler didnt find any cell in plate {plate_i} well {well_i}')
                    missed_cells += 1
                    continue

                if np.sqrt(dist_2) > 35: #CellProfiler probably failed to find that cell, ignore
                    print(f'One GT cell has been ignored of class {row["GT_label"]}')
                    class_miss[row['GT_label']-1]+=1
                    missed_cells += 1
                    continue

                label = row['GT_label']
                ### IGNORE CELL OF CLUSTER 7 AND 8 (Cell segmentation failed and too poorly represented)
                if (label==7 or label==8):
                    continue
                unique_id = "CellClass_{}_{}_{}_id{}.tiff".format(label,plate_i,well_i,index)
                #Store all useful info about that cell to link its GT and cellprofiler metadata
                new_row=pd.Series([plate_i,well_i,row['GT_label'],unique_id,row['GT_pos_x'],row['GT_pos_y'],CP_Pi_Wi.loc[CP_row_min,'ImageNumber'],CP_Pi_Wi.loc[CP_row_min,'ObjectNumber'],CP_Pi_Wi.loc[CP_row_min,'AreaShape_Center_X'],CP_Pi_Wi.loc[CP_row_min,'AreaShape_Center_Y']], index=MetaData_GT_link_CP.columns)
                MetaData_GT_link_CP = MetaData_GT_link_CP.append(new_row, ignore_index=True)


                #save a 3 Channel tiff file per single cell, in a folder corresponding to GT class
                blue_path = path_to_CP+'SingleWholeCellCroppedImages/CroppedImages_Blue/'
                green_path = path_to_CP+'SingleWholeCellCroppedImages/CroppedImages_Green/'
                red_path = path_to_CP+'SingleWholeCellCroppedImages/CroppedImages_Red/'

                well_site_folder = f'{synth_plate}_{well_i}/'
                object_num = CP_Pi_Wi.loc[CP_row_min,'ObjectNumber']
                single_cell_file = f'SplitCellBodies_{object_num}.png'

                img_blue = io.imread(blue_path+well_site_folder+single_cell_file)
                img_green = io.imread(green_path+well_site_folder+single_cell_file)
                img_red = io.imread(red_path+well_site_folder+single_cell_file)

                #new stacked RGB img
                img_rgb = np.stack([img_red,img_green,img_blue],axis=-1)

                folder_name = list_folder[row['GT_label']-1]
                io.imsave(save_folder+f'{folder_name}/'+unique_id,img_rgb,plugin='tifffile')

MetaData_GT_link_CP.to_csv('DataSets/MetaData2_PeterHorvath_GT_link_CP.csv')

print('All files processed')
print(f'{missed_cells} GT cells were not detected by cell profiler')
print(f'A subsample of {len(MetaData_GT_link_CP)} cells was created')

#184 cell were missed by cell profiler
# 15 min of run for 8 synth plate over 32
# 77'910 cells in total
# class miss was : [  1. 140.  14.   0.   5.   0.   8.  16.]
#print(class_miss)
#3h of run for all 32 plate
#735 cells has been not detected
#310194 Cells in total
#No cells in plate 29 M20
#class miss_was [  5. 506.  70.   0.  24.   0.  70.  54.]
#%%

#########################################################
############# Statistics on the dataset 2 ###############
#########################################################

##### AND PRODUCTION OF A SUBSAMPLE AS IT HAS BEEN FINALLY USED

Metadata1 = pd.read_csv('DataSets/MetaData2_PeterHorvath_GT_link_CP.csv')

#Take a subset of the whole dataset
Metadata1 = Metadata1.groupby('GT_label').apply(lambda x: x.sample(n=2500 if len(x)>2500 else len(x)))

cluster_count = []
for i in range(6):
    value = Metadata1[Metadata1['GT_label']==i+1]['Unique_ID'].count()
    cluster_count.append(value)

cluster_count = np.array(cluster_count)
total_cell = np.sum(np.array(cluster_count))
cluster_count
cmap1 = plt.get_cmap('tab20')
colors1 = cmap1(np.linspace(0,1.0,20))
cmap2 = plt.get_cmap('Set3')
colors2 = cmap2(np.linspace(0,1.0,10))
colors = np.concatenate((colors1,colors2),0)
plt.figure(figsize=(12,8))
plt.bar(range(1,7),cluster_count,color=colors[[0,1,2,3,4,5]])
for i, pos in enumerate(cluster_count):
    plt.text(i+1,pos+30,str(pos)+' ('+str(np.round(pos/np.sum(cluster_count)*100,2))+'%)',ha='center',color='black',fontweight='bold')
plt.title(f'Dataset 2 cluster distribution ({len(Metadata1)} cells in total)')
# plt.bar(range(1,9),class_miss,color=colors[[0,1,2,3,4,5,6,7]])
# for i, pos in enumerate(class_miss):
#     plt.text(i+1,pos+2,str(pos),ha='center',color='black',fontweight='bold')
# plt.title(f'Single cells not detected by Cell Profiler ({np.sum(class_miss)} cells in total)')
plt.ylabel('# of cells')
plt.xlabel('Cluster')


#%%
###########################################
##### Extract the files from the subsample
###########################################
# from shutil import copy2
#
# origin_folder = 'DataSets/Peter_Horvath_Data/'
# destin_folder = 'DataSets/Peter_Horvath_Subsample/'
#
# list_class = ['Class_'+str(i) for i in range(1,7)]
#
# # From the whole dataset (300k cells), extract the files of the subsample
# subsample_df = pd.read_csv('DataSets/MetaData2_PeterHorvath_GT_link_CP_SUBSAMPLE.csv')
# for index, row in subsample_df.iterrows():
#     label = row['GT_label']
#     file_name = row['Unique_ID']
#
#     src= origin_folder+list_class[int(label-1)]+'/'+file_name
#     dst= destin_folder+list_class[int(label-1)]
#     copy2(src,dst)
#
