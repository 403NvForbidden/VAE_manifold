# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-04-09T10:37:42+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-04-09T18:11:57+10:00
'''
CellProfiler output single_cell images (from segmentation) in a given folder architecture :
a separate folder for each Well_XY_channel
This code enable to regroupe images of same Well and XY in one HxWx4 tif file
All channel are already in the same shape
'''


import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.external import tifffile

root_path = 'single_cell_image/'
saving_path = 'single_cell_processed/'
non_matching_files = 0

#All the full_size_images
all_subfolders = [f for f in os.listdir(root_path) if not f.startswith('.')]

counter = 0
for subfolder in all_subfolders:
    # name of subfolder are of type WellXX_SlideXX_CondtionXX_XYxx_ChannelXX

    infos = subfolder.split('_')

    #Consider only subfolder with channel1 (regroup image only one time)
    if infos[4] == 'Channel1':
        counter += 1
        print(f'Processing unique condition {counter} over {len(all_subfolders)/4}')
        channels = ['Channel1','Channel2','Channel3','Channel4']
        sep = '_'
        name = f'{sep.join(infos[0:4])}'

        #Overwrite saving folder if it alreadz exist
        saving_folder = f'{saving_path}{name}'
        if os.path.exists(saving_folder):
            shutil.rmtree(saving_folder)
        os.makedirs(saving_folder)

        #loop over each single_cell_file for that specific condition
        c = 0
        #Channel1 folder exist for sure because we entered the if statement
        for cell_bodies in [f for f in os.listdir(f'{root_path}{name}_{channels[0]}') if not f.startswith('.')]:
            c += 1

            ##Sometimes all 4channel does not exist for a given cel; -> count that error but continue
            try:

                #All for channel of each diff cell are stack together
                array_list = [io.imread(f'{root_path}{name}_{channel}/{cell_bodies}') for channel in channels]

                final_array = np.stack(array_list,axis=2)  #of shape HxWx4

                #SAVE THE IMAGE (in tiff because 4 channel)
                io.imsave(f'{saving_folder}/cell{c}.tif',final_array)

            except FileNotFoundError:
                non_matching_files += 1
                continue

print(f'{non_matching_files} files were not found in all 4 channels....')
