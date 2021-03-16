# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-07-24T19:33:06+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-31T10:54:00+10:00

'''
Quick analysis of the shape ditribution of the single cell images of different datasets
'''

import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

#dataset_folder = '/Chaffer_Data/'
dataset_folder = '/home/sachahai/Documents/VAE_manifold/DataSets/Felix_Full_128/'
#dataset_folder = '/home/sachahai/Documents/VAE_manifold/DataSets/Felix_Full_Complete/'

all_class_subfolder = [f for f in os.listdir(dataset_folder) if not f.startswith('.')]
#Store shape -> Tuple
shapes = []
for class_subfoler in all_class_subfolder:
    path_to_files = os.path.join(dataset_folder, class_subfoler)
    if os.path.isdir(path_to_files):
        all_full_imgs = [f for f in os.listdir(path_to_files) if not f.startswith('.')]
        for singe_cell in all_full_imgs:
            img = io.imread(os.path.join(path_to_files, singe_cell))
            shapes.append(img.shape)

# %% Play with the shapes
import seaborn
seaborn.set()

mean = np.mean([shapex[0] for shapex in shapes])
median = np.median([shapex[0] for shapex in shapes])

plt.figure(figsize=(10,6),dpi=300)
seaborn.distplot([shapex[0] for shapex in shapes],hist=False,kde=True, rug=True, label='Height')
seaborn.distplot([shapex[1] for shapex in shapes],hist=False,kde=True, rug=True,  label='Width')
plt.xlabel('#pixels')
plt.ylabel('frequencies')
plt.title(f"Single cell image size distribution of Felix dataset, over {len(shapes)} images")
plt.axvline(median,ls='--',label='median')
plt.axvline(64,lw=2,color='r',label=f'FIXED SIZE {len([shapex[0] for shapex in shapes if shapex[0] < 64]) / len(shapes):.2f}%')
plt.legend()
plt.show()
#plt.savefig(os.path.join(dataset_folder, 'dataset2_distribution.png'))

# %% Analysis of height vs width ratio
h = np.array([shapex[0] for shapex in shapes])
w = np.array([shapex[1] for shapex in shapes])
seaborn.distplot(h/w,rug=True)
plt.xlabel('width ratio')
plt.ylabel('height ratio')
plt.title('Dataset 1, image ratio distribution')
plt.show()