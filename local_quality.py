# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-06-17T17:34:09+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-06-18T16:30:06+10:00

'''
This file reproduces the work from paper "Howto Evaluate Dimensionality Reduction ? -
Improving the Co-ranking Matrix" from Michael Biehl and Barbara Hammer

Basically, it computes a LCMC with  theco-ranking matrix,
but having two distinct parameter rather than only one

Param 1 :
ks, defines the considered neighborhood size (former K in LCMC). Rank higher
than ks are not considered in the metric as they are considered as not relevent

Param 2 :
kt, rank error that are tolerated. Standard LCMC will consider as sucess all points
that are still in neighborhood K, but here we want the rank error to be smaller than
kt to be considered as a success (the occurence should lie on an off-diagonal < kt on
the co-ranking matrix)
'''

import numpy as np
from scipy.spatial import distance

def wt(rho_ij,r_ij,kt):
    '''
    Tolerance function of the qualitative assessment

    rho_ij : rank_matrix in which element i,j is rank of point j w.r.t i in high dim space
    r_ij : rank matrix in which element i,j is rank of point j w.r.t i in the projection
    kt : tolerance parameter (= accepted rank error)

    matrice where element i,j is 1 if the pair of point i and j in projection is considered as a success
    '''

    mat_res = np.ones_like(rho_ij)

    mat_res[np.abs(rho_ij-r_ij) > kt]=0

    return mat_res

def ws(rho_ij,r_ij,ks):
    '''
    Determine if the points and its projection will be taken in
    account in the metric. If rank is higher than the neighborhood size ks in
    both input and output the point can't be considered as a success.
    higher rank are judged as less important.

    Return a Matrix where each element i,j are 0 if the pair i_j is a success
    '''

    mat_res = np.ones_like(rho_ij)

    mat_res[np.logical_and((rho_ij>ks),(r_ij>ks))]=0

    return mat_res

def local_quality(high_data, low_data, kt, ks):
    '''
    :param high_data: DataFrame containing the higher dimensional data.
    :param low_data: DataFrame containing the lower dimensional data.
    :returns: a score for each data_point that express the local quality. Data
    points are kept in the same order than in input
    '''

    n, m = high_data.shape
    #generated two n by n distance matrix. Data points stays in same order
    high_distance = distance.squareform(distance.pdist(high_data))
    low_distance = distance.squareform(distance.pdist(low_data))

    #np.argsort return the indices that would sort an array
    # -. argsort two times to obtain the rank matrix
    high_ranking = high_distance.argsort(axis=1).argsort(axis=1)
    low_ranking = low_distance.argsort(axis=1).argsort(axis=1)
    #one line in high_ranking is a rho_ij for a fixed i and all j

    ws_mat = ws(high_ranking,low_ranking,ks)
    wt_mat = wt(high_ranking,low_ranking,kt)

    local_quality_score = 1./(2*ks*n) * (np.sum(ws_mat*wt_mat,axis=1) + (np.sum(ws_mat*wt_mat,axis=0)))
    # len = n , one score for each data point that correspond to the local quality score
    return local_quality_score



# %% TOY EXAMPLE
# from sklearn.datasets import make_swiss_roll
# from sklearn.manifold import TSNE
# X, color = make_swiss_roll(n_samples=1500)
# X_embedded = TSNE(n_components=2,perplexity=50).fit_transform(X)
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# Axes3D
#
# fig = plt.figure(figsize=(8,12))
# ax = fig.add_subplot(211, projection='3d')
# ax.scatter(X[:,0],X[:,1],X[:,2], c=color, cmap=plt.cm.Spectral)
#
# ax.set_title("Original Data")
# ax = fig.add_subplot(212)
# ax.scatter(X_embedded[:,0],X_embedded[:,1],c=color, cmap=plt.cm.Spectral)
# plt.axis('tight')
# plt.xticks([]), plt.yticks([])
# plt.title('projected data')
# plt.show()
#
# #%%
# kt = 700
# ks = 700
# local_quality_score = local_quality(X, X_embedded, kt=kt, ks=ks)
#
# fig = plt.figure(figsize=(8,6))
#
# ax.set_title("Projection Colored by Local quality contribution")
# ax = fig.add_subplot(111)
# bar = ax.scatter(X_embedded[:,0],X_embedded[:,1],c=local_quality_score, cmap=plt.cm.cool_r)
# cbar = fig.colorbar(bar)
# plt.axis('tight')
# plt.xticks([]), plt.yticks([])
# plt.title(f"Projection Colored by Local quality contribution - kt : {kt}, ks = {ks}")
# plt.show()
