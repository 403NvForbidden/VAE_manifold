# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-05-08T11:28:59+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-05-13T16:10:16+10:00



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import os
import shutil

CPpath = 'DataSets/CellProfiler_Outputs/'
CPQuantitativeFiles ='Quantitative_Outputs/'
CPcsv = '200507_Horvath_Synth_Simple_Cell_Profiler_AnalysisWholeCell.csv'


# Extraction of info from CellProfiler MetaData

fields = ['ImageNumber','ObjectNumber','Metadata_Site','Metadata_Well','AreaShape_Center_X','AreaShape_Center_Y']

CP_df = pd.read_csv(CPpath+CPQuantitativeFiles+CPcsv,usecols=fields)

# %%

CP_df.head()
CP_df.keys()
CP_df.shape

# %%
img_test_path = 'DataSets/BBBC031_v1_dataset/Images/ProcessPlateSparse_wA01_s01_z1_t1_CELLMASK.png'
img = io.imread(img_test_path)

fig, ax = plt.subplots(1,1,figsize=(10, 10))
ax.imshow(img)

site1 = CP_df['Metadata_Site']==1
Well1 = CP_df['Metadata_Well']=='A01'

#for index, row in CP_df[site1 & Well1][['AreaShape_Center_X','AreaShape_Center_Y']].iterrows() :
    #print(row['AreaShape_Center_X'])
#no need for a loop

x = CP_df[site1 & Well1]['AreaShape_Center_X']
#print(x.values)
y = CP_df[site1 & Well1]['AreaShape_Center_Y']

ax.scatter(x.values,y.values,s=60,c='red',marker='X')


# %% Extraction of info from BBBC Groundtruh

BBBC_GT_path = 'DataSets/BBBC031_v1_DatasetGroundTruth.csv'

fields = ['ImageName','CellIdx','LocationX','LocationY','ProcessID','ColorParamsR','ColorParamsG','ColorParamsB','ShapeParams']

GT_df = pd.read_csv(BBBC_GT_path,sep=';',usecols=fields)

GT_df.shape
Well = []
Site = []
# %% Add columns of well and site
for name in GT_df['ImageName'].values :
    subparts = name.split('_')
    Well.append(subparts[1][1:])
    Site.append(int(subparts[2][-1]))

## Add the columns to GT dataframe

GT_df['Well'] = Well
GT_df['Site'] = Site
GT_df.head()


# %% From GT BBBC, find for all cells the closest CellProfiler centroid
# Save the merge RGB tiff file in a folder corresponding to the GT process (label)
# Save an appropriate meta data

def closest_point(point,points):
    points = np.asarray(points)
    dist_2 = np.sum((points-point)**2, axis=1)
    return np.argmin(dist_2), dist_2[np.argmin(dist_2)]


#Folder were to save data
save_folder = 'DataSets/Synthetic_Data_1/'
list_folder = ['Process_1','Process_2','Process_3','Process_4','Process_5','Process_6','Process_7']
#for folder in list_folder:
    #saving_folder = f'{save_folder}{folder}'
    #if os.path.exists(saving_folder):
        #shutil.rmtree(saving_folder)
    #os.makedirs(saving_folder)

MetaData_GT_link_CP = pd.DataFrame(columns=['Well','Site','GT_label','GT_Cell_id','Unique_ID','GT_x','GT_y','GT_colorR','GT_colorG','GT_colorB','GT_Shape','CP_ImagerNumber','CP_ObjectNumber','CP_x','CP_y'])

counter = 0

#Iterate over images (combination of well and site)
list_of_well_site = os.listdir('DataSets/CellProfiler_Outputs/SingleWholeCellCroppedImages/SingleWholeCellCroppedImages_Blue')
for combination in list_of_well_site:
    strings = combination.split('_')
    well_i = strings[0]
    site_i = int(strings[1][-1])


    #CellProfiler Well 1 site 1 list of possible centroid
    sitei_CP = CP_df['Metadata_Site']==site_i
    Welli_CP = CP_df['Metadata_Well']==well_i
    CP_WA1_S1 = CP_df[sitei_CP & Welli_CP].reset_index()
    CP_centroids_array = CP_WA1_S1[['AreaShape_Center_X','AreaShape_Center_Y']].values
    # For know we only focus on well 1 site 1
    sitei_GT = GT_df['Site']==site_i
    Welli_GT = GT_df['Well']==well_i

    #iterate over each single cell of a given well-site combination
    for index, row in GT_df[sitei_GT & Welli_GT].iterrows():
        GT_centroid = np.array([row['LocationX'],row['LocationY']])

        #Find the closest cell profiler centroid
        CP_row_min, dist_2 = closest_point(GT_centroid,CP_centroids_array)
        CP_centroid = CP_centroids_array[CP_row_min]

        if np.sqrt(dist_2) > 30: #CellProfiler probably failed to find that cell, ignore
            print('One GT cell has been ignored')
            counter += 1
            continue

        label = row['ProcessID']
        id = row['CellIdx']
        well_site = combination
        unique_id = "CellProcess_{}_{}_id{}.tiff".format(label,combination,id)
        #Store all useful info about that cell to link its GT and cellprofiler metadata
        new_row = pd.Series([row['Well'],row['Site'],row['ProcessID'],row['CellIdx'],unique_id,row['LocationX'],row['LocationY'],row['ColorParamsR'],row['ColorParamsG'],row['ColorParamsB'],row['ShapeParams'],CP_WA1_S1.loc[CP_row_min,'ImageNumber'],CP_WA1_S1.loc[CP_row_min,'ObjectNumber'],CP_WA1_S1.loc[CP_row_min,'AreaShape_Center_X'],CP_WA1_S1.loc[CP_row_min,'AreaShape_Center_Y']], index=MetaData_GT_link_CP.columns)
        MetaData_GT_link_CP = MetaData_GT_link_CP.append(new_row, ignore_index=True)
        #save one 3 Channel tiff file per single cell, in a folder corresponding to GT
        blue_path = 'DataSets/CellProfiler_Outputs/SingleWholeCellCroppedImages/SingleWholeCellCroppedImages_Blue/'
        green_path = 'DataSets/CellProfiler_Outputs/SingleWholeCellCroppedImages/SingleWholeCellCroppedImages_Green/'
        red_path = 'DataSets/CellProfiler_Outputs/SingleWholeCellCroppedImages/SingleWholeCellCroppedImages_Red/'

        w = row['Well']
        r = row['Site']
        well_site_folder = f'{w}_0{r}/'
        object_num = CP_WA1_S1.loc[CP_row_min,'ObjectNumber']
        single_cell_file = f'WholeCell_{object_num}.png'

        #img_blue = io.imread(blue_path+well_site_folder+single_cell_file)
        #img_green = io.imread(green_path+well_site_folder+single_cell_file)
        #img_red = io.imread(red_path+well_site_folder+single_cell_file)

        #new stacked RGB img
        #img_rgb = np.stack([img_red,img_green,img_blue],axis=-1)

        ## TODO: STACK DIFFERENT CHANNEL
        #file_name = "CellProcess_{}_{}_id{}.tiff".format(label,combination,id)
        #folder_name = list_folder[row['ProcessID']-1]
        #io.imsave(save_folder+f'{folder_name}/'+file_name,img_rgb,plugin='tifffile')


MetaData_GT_link_CP.head()
MetaData_GT_link_CP.shape

MetaData_GT_link_CP.to_csv('DataSets/MetaData1_GT_link_CP.csv')

print('All files processed')
print(f'{counter} GT cells were not detected by cell profiler')

#%%
############# Statistics on the dataset 1 ###############
#########################################################
Metadata1 = pd.read_csv('DataSets/MetaData1_GT_link_CP.csv')

cluster_count = []
for i in range(7):
    value = Metadata1[Metadata1['GT_label']==i+1]['GT_Cell_id'].count()
    cluster_count.append(value)

cluster_count = np.array(cluster_count)
total_cell = np.sum(np.array(cluster_count))

cmap1 = plt.get_cmap('tab20')
colors1 = cmap1(np.linspace(0,1.0,20))
cmap2 = plt.get_cmap('Set3')
colors2 = cmap2(np.linspace(0,1.0,10))
colors = np.concatenate((colors1,colors2),0)

plt.bar(range(1,8),cluster_count,color=colors[[0,1,2,3,4,5,6]])
for i, pos in enumerate(cluster_count):
    plt.text(i+1,pos-200,str(pos),ha='center',color='black',fontweight='bold')
plt.title('Dataset 1 cluster distribution')
plt.ylabel('# of cells')
plt.xlabel('Cluster')
# %%

img = io.imread('DataSets/BBBC031_v1_dataset/Images/ProcessPlateSparse_wA02_s01_z1_t1_CELLMASK.png')
fig, ax = plt.subplots(1,1,figsize=(10, 10))
ax.imshow(img)

site1 = GT_df['Site']==1
Well1 = GT_df['Well']=='A02'

for index, row in GT_df[site1 & Well1].iterrows():
    ax.text(row['LocationX'],row['LocationY'],row['ProcessID'],backgroundcolor='w')



# %%
Metadata1 = pd.read_csv('DataSets/MetaData1_GT_link_CP.csv')
Metadata1.head()


Metadata1.sort_values(by=['Unique_ID']).head()
Metadata1.sort_values(by=['Unique_ID']).head().index
test = Metadata1.Unique_ID.values
test2 = Metadata1.Unique_ID.values
np.all(test == test2)


temp_matching_df = pd.DataFrame(columns=['x_coord','y_coord','z_coord','Unique_ID'])
temp_matching_df.x_coord=[1,2,3]
temp_matching_df.head()

temp_matching_df['asdasd'] = [5,456,456]
temp_matching_df.head()
temp_matching_df[['te','sfdgdf','45345435']]=[[1,4,5],[4,5,7],[4,5,6]]

#%%
import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch import cuda
from torchvision.utils import save_image, make_grid
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.colors
import matplotlib.cm
import itertools
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plotly.express as px


import sys, os
import plotly.offline
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication

GT = pd.read_csv('DataSets/BBBC031_v1_DatasetGroundTruth.csv',sep=';')


def show_in_window(fig):


    plotly.offline.plot(fig, filename='name.html', auto_open=False)

    app = QApplication(sys.argv)
    web = QWebEngineView()
    file_path = os.path.abspath(os.path.join(os.getcwd(), "name.html"))
    web.load(QUrl.fromLocalFile(file_path))
    web.show()
    sys.exit(app.exec_())

fig_test = px.scatter(GT, x='PositionOnRegressionPlaneX', y='PositionOnRegressionPlaneY',color='ProcessID')

#fig_test.write_image('GT_latentCode.png')
show_in_window(fig_test)
#from plotly.offline import init_notebook_mode, plot_mpl
#plotly.io.orca.config.executable = '/home/sachahai/miniconda2/pkgs/plotly-orca-1.2.1-1/bin/orca'
#plotly.io.orca.config.save()