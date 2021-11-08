#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:01:24 2021

@author: sachahai
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVC
import pandas as pd
import plotly

import pickle as pkl
import os

from helpers import metadata_latent_space, plot_from_csv
from scipy import spatial

with open(r"/mnt/Linux_Storage/outputs/vaDE/backbone_metric/backbone_points.pkl", "rb") as input_file:
    e = pkl.load(input_file)

GT_df = pd.read_csv("/mnt/Linux_Storage/outputs/vaDE/embeded_data.csv")
GT_df = GT_df[GT_df['GT_label'] != 7]
tree = spatial.KDTree(GT_df[['z0', 'z1', 'z2']])

# convert to data frames
array_list = []
df = pd.DataFrame()
for index, coords in enumerate(e):
    for c in coords:
        array_list.append(tuple(c))

uniques = np.unique(array_list,  axis=0)

result_index = []
for query in uniques:
    result_index.append(tree.query(query, k = 3)[1])
result_index = [item for sublist in result_index for item in sublist]

result_df=GT_df.iloc[result_index]


figplotly = plot_from_csv(result_df, low_dim_names=['z0', 'z1', 'z2'], dim=3, as_str=True)
plotly.offline.plot(figplotly, filename=os.path.join("/mnt/Linux_Storage/outputs/vaDE/backbone_metric", 'test.html'), auto_open=True)


import shutil

# find images via unique id
for index, row in result_df.iterrows():
    src = f"/home/sachahai/Documents/VAE_manifold/DataSets/Synthetic_Data_1/Process_{row.GT_label}/{row.Unique_ID}"
    dst = f"/home/sachahai/Documents/VAE_manifold/DataSets/Synthetic_Data_1/subset/Process_{row.GT_label}/{row.Unique_ID}"
    shutil.copyfile(src, dst)

result_df.to_csv("/mnt/Linux_Storage/outputs/1_experiment/zzzz.csv", index=False)