# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-05T10:19:19+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahaidinger
# @Last modified time: 2020-04-06T09:38:35+10:00

'''
Quick analysis of the shape ditribution of the single cell images
'''

import os
import matplotlib.pyplot as plt
import numpy as np

from skimage import io

root_path = 'single_cell_image/'

#Store shape -> Tuple
shapes = []

#All the full_size_images
all_full_imgs = [f for f in os.listdir(root_path) if not f.startswith('.')]

# %% Enter each subFolder
c = 0
for full_img in all_full_imgs:
    if c%5 == 0:
        print(f'File number {c} over {len(all_full_imgs)}')
    all_single_cell_file = [f for f in os.listdir(f'{root_path}{full_img}') if not f.startswith('.')]

    for singe_cell in all_single_cell_file:
        img = io.imread(os.path.join(f'{root_path}{full_img}/{singe_cell}'))
        shapes.append(img.shape)

# %% Play with the shapes
import seaborn
seaborn.set()

mean = np.mean([shapex[0] for shapex in shapes])
median = np.median([shapex[0] for shapex in shapes])

plt.figure(figsize=(14,8))
seaborn.distplot([shapex[0] for shapex in shapes],hist=False,kde=True, rug=True, label='Height')
seaborn.distplot([shapex[1] for shapex in shapes],hist=False,kde=True, rug=True,  label='Width')
plt.xlabel('#pixels')
plt.ylabel('frequencies')
plt.title(f'Shape distribution of single cell images, over {len(shapes)} images')
plt.axvline(median,ls='--',label='median')
plt.axvline(128,lw=2,color='r',label='FIXED SIZE')
plt.legend()
plt.show()
# %%
h = np.array([shapex[0] for shapex in shapes])
w = np.array([shapex[1] for shapex in shapes])
seaborn.distplot(h/w,rug=True)
