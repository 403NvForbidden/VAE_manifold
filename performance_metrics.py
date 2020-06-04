# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-05-25T11:37:48+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-06-04T22:59:38+10:00

'''
File containing different metrics that are used to evaluate the
latent space quality
'''

import pandas as pd
import numpy as np
from classifier_net import Dataset_from_csv, Classifier_Net, train_net, perf_eval, perf_eval_perclass
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn import metrics
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline
from helpers import plot_from_csv
from data_processing import get_inference_dataset
from MINE_net import MINE, train_MINE
from scipy import stats




def classifier_performance(path_to_csv,Metrics=[True,False,False],num_iteration=5):
    '''
    Given a CSV-file containing a 3D latent code to evaluate, built a simple
    (200 unit single hidden layer) NN classifier. Test accuracy can be used as
    a numerical value to assess the quality of the latent code.
    High accuracy -> the latent dimensions present the data in a way where ground
    truth cluster can easily be discriminate

    Metric 1 : Test classification performance on all single cell except uniform cluster (7)
    Metric 2 : Same, but only strong phenotype change (>0.5)
    Metric 3 : Same, but meta-cluster. Discriminate between 1&2, 3&4 and 5&6
    '''
    #################################
    ####### Metric One ##############
    #################################
    ## All cells included, except cluster 7
    if Metrics[0]:
        latentCode_frame = pd.read_csv(path_to_csv)
        non_last_cluster = latentCode_frame['GT_label'] != 7
        latentCode_frame = latentCode_frame[non_last_cluster]
        #For NN, the class must be between 0 - num_class-1
        latentCode_frame['GT_label'] = latentCode_frame['GT_label'].subtract(1)

        ##Make a half / half train test split that have the same percentage of classes as original dataset
        labels = latentCode_frame['GT_label'].values
        train_test_split = StratifiedShuffleSplit(n_splits=num_iteration,test_size=0.5,random_state=12) #Change n_splits if want to have several run

        train_acc = []
        test_acc = []
        perclass_te_acc = []
        #For statistical relevance, make several train-test split
        for train_index, test_index in train_test_split.split(np.zeros((len(latentCode_frame),3)),labels):
            dataset_train = latentCode_frame.iloc[train_index]
            dataset_test = latentCode_frame.iloc[test_index]
            dataset_train.reset_index(inplace=True)
            dataset_test.reset_index(inplace=True)

            #Built train and test dataset/dataloader
            tr_dataset = Dataset_from_csv(dataset_train,'GT_label')
            tr_dataloader = DataLoader(tr_dataset,batch_size=128,shuffle=True)
            te_dataset = Dataset_from_csv(dataset_test,'GT_label')
            te_dataloader = DataLoader(te_dataset,batch_size=128,shuffle=True)

            model_1 = Classifier_Net()
            model_1 = model_1.float()
            model_1 = model_1.cuda()

            #train on train_dataloader
            train_net(model_1,20,tr_dataloader)

            #train and test accuracy
            train_acc.append(perf_eval(model_1,tr_dataloader))
            test_acc.append(perf_eval(model_1,te_dataloader))

            perclass_te_acc.append(perf_eval_perclass(model_1,te_dataloader))

        return train_acc, test_acc, perclass_te_acc

    #################################
    ####### Metric two ##############
    #################################
    ## Disregard cluster 7, and only consider strong phenotypic change (>0.5)
    if Metrics[1]:
        latentCode_frame = pd.read_csv(path_to_csv)
        non_last_cluster = latentCode_frame['GT_label'] != 7
        strong_phenotype = latentCode_frame['GT_dist_toMax_phenotype']>=0.5
        latentCode_frame = latentCode_frame[non_last_cluster & strong_phenotype]
        #For NN, the class must be between 0 - num_class-1
        latentCode_frame['GT_label'] = latentCode_frame['GT_label'].subtract(1)

        ##Make a half / half train test split that have the same percentage of classes as original dataset
        labels = latentCode_frame['GT_label'].values
        train_test_split = StratifiedShuffleSplit(n_splits=num_iteration,test_size=0.5,random_state=12) #Change n_splits if want to have several run

        train_acc = []
        test_acc = []
        perclass_te_acc = []
        #For statistical relevance, make several train-test split
        for train_index, test_index in train_test_split.split(np.zeros((len(latentCode_frame),3)),labels):
            dataset_train = latentCode_frame.iloc[train_index]
            dataset_test = latentCode_frame.iloc[test_index]
            dataset_train.reset_index(inplace=True)
            dataset_test.reset_index(inplace=True)

            #Built train and test dataset/dataloader
            tr_dataset = Dataset_from_csv(dataset_train,'GT_label')
            tr_dataloader = DataLoader(tr_dataset,batch_size=128,shuffle=True)
            te_dataset = Dataset_from_csv(dataset_test,'GT_label')
            te_dataloader = DataLoader(te_dataset,batch_size=128,shuffle=True)

            model_2 = Classifier_Net()
            model_2 = model_2.float()
            model_2 = model_2.cuda()

            #train on train_dataloader
            train_net(model_2,20,tr_dataloader)

            #train and test accuracy
            train_acc.append(perf_eval(model_2,tr_dataloader))
            test_acc.append(perf_eval(model_2,te_dataloader))

            perclass_te_acc.append(perf_eval_perclass(model_2,te_dataloader))

        return train_acc, test_acc, perclass_te_acc

    #################################
    ####### Metric three ##############
    #################################
    ## Disregard cluster 7, only strong phenotypic change (>0.5), and METACLUSTER
    # 1&2 vs 3&4 vs 5&6
    if Metrics[2]:
        latentCode_frame = pd.read_csv(path_to_csv)
        non_last_cluster = latentCode_frame['GT_label'] != 7
        strong_phenotype = latentCode_frame['GT_dist_toMax_phenotype']>=0.5
        latentCode_frame = latentCode_frame[non_last_cluster & strong_phenotype]
        #For NN, the class must be between 0 - num_class-1
        latentCode_frame['GT_label'] = latentCode_frame['GT_label'].subtract(1)

        #Built metacluster (1&2 vs 3&4 vs 5&6)
        latentCode_frame['GT_label'].replace(1,0,inplace=True)
        latentCode_frame['GT_label'].replace(2,1,inplace=True)
        latentCode_frame['GT_label'].replace(3,1,inplace=True)
        latentCode_frame['GT_label'].replace(4,2,inplace=True)
        latentCode_frame['GT_label'].replace(5,2,inplace=True)

        ##Make a half / half train test split that have the same percentage of classes as original dataset
        labels = latentCode_frame['GT_label'].values
        train_test_split = StratifiedShuffleSplit(n_splits=num_iteration,test_size=0.5,random_state=12) #Change n_splits if want to have several run

        train_acc = []
        test_acc = []
        perclass_te_acc = []
        #For statistical relevance, make several train-test split
        for train_index, test_index in train_test_split.split(np.zeros((len(latentCode_frame),3)),labels):
            dataset_train = latentCode_frame.iloc[train_index]
            dataset_test = latentCode_frame.iloc[test_index]
            dataset_train.reset_index(inplace=True)
            dataset_test.reset_index(inplace=True)

            #Built train and test dataset/dataloader
            tr_dataset = Dataset_from_csv(dataset_train,'GT_label')
            tr_dataloader = DataLoader(tr_dataset,batch_size=128,shuffle=True)
            te_dataset = Dataset_from_csv(dataset_test,'GT_label')
            te_dataloader = DataLoader(te_dataset,batch_size=128,shuffle=True)

            model_3 = Classifier_Net(num_of_class=3)
            model_3 = model_3.float()
            model_3 = model_3.cuda()

            #train on train_dataloader
            train_net(model_3,20,tr_dataloader)

            #train and test accuracy
            train_acc.append(perf_eval(model_3,tr_dataloader))
            test_acc.append(perf_eval(model_3,te_dataloader))

            perclass_te_acc.append(perf_eval_perclass(model_3,te_dataloader))

        return train_acc, test_acc, perclass_te_acc


def compare_models(list_of_csv,Metrics=[True,False,False],num_iteration=5):

    model_names=[]
    all_means=[]
    all_stds=[]

    for i, csv_file in enumerate(list_of_csv):

        train_acc, test_acc, perclass_te_acc = classifier_performance(csv_file,Metrics,num_iteration)

        #In future, probably don-t care about test accuracy
        all_means.append(np.mean(train_acc))
        all_stds.append(np.std(train_acc))
        model_names.append(f'Model {i+1} -Train')
        all_means.append(np.mean(test_acc))
        all_stds.append(np.std(test_acc))
        model_names.append(f'Model {i+1} -Test')


        perclass_means = np.mean(np.array(perclass_te_acc),axis=0)
        perclass_stds = np.std(np.array(perclass_te_acc),axis=0)

        #One plot per model for the per-class accuracy
        names = [f'cluster {j}' for j in range(1,len(perclass_means)+1)]
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(names)),perclass_means,yerr=perclass_stds,align='center',alpha=0.5, ecolor='black',capsize=10)
        ax.set_ylabel('Test classification accuracy [%]')
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names)
        ax.set_title(f'Per Class accuracy of model {i+1}, avg over 10 runs')
        ax.yaxis.grid(True)

        plt.tight_layout()
        #plt.savefig()
        plt.show()

    fig2, ax2 = plt.subplots(figsize=(10,6))
    ax2.bar(np.arange(len(model_names)),all_means,yerr=all_stds,align='center',alpha=0.5, ecolor='black',capsize=10)
    ax2.set_ylabel('Classification accuracy [%]')
    ax2.set_xticks(np.arange(len(model_names)))
    ax2.set_xticklabels(model_names)
    ax2.set_title('NN Classifier acc (avg over 10 runs) as latent code performance metric')
    ax2.yaxis.grid(True)

    plt.tight_layout()
    #plt.savefig()
    plt.show()



def closest_point(point,points,dim=2):
    '''
    Compute the eucl. distance between a point and the closest points in a list of points
    return both the distance and the indice of the closests point
    '''
    points = np.asarray(points)
    dist_2 = np.sum((points-point)**2, axis=1)
    if dim==2:
        return np.argmin(dist_2), np.sqrt(dist_2[np.argmin(dist_2)])
    if dim==3:
        return np.argmin(dist_2), dist_2[np.argmin(dist_2)]**(1./3.)

def sqr_distance(point1,point2):
    '''
    Compute the eucl. distance between a point and an other points
    '''
    p1 = np.array(point1)
    p2 = np.array(point2)
    sqr_dist = np.sum((p1-p2)**2, axis = 0)
    dist = np.sqrt(sqr_dist)

    return dist

import math

def dot(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z

def length(v):
    x,y,z = v
    return math.sqrt(x*x + y*y + z*z)

def vector(b,e):
    x,y,z = b
    X,Y,Z = e
    return (X-x, Y-y, Z-z)

def unit(v):
    x,y,z = v
    mag = length(v)
    return (x/mag, y/mag, z/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y,z = v
    return (x * sc, y * sc, z * sc)

def add(v,w):
    x,y,z = v
    X,Y,Z = w
    return (x+X, y+Y, z+Z)


# Given a line with coordinates 'start' and 'end' and the
# coordinates of a point 'pnt' the proc returns the shortest
# distance from pnt to the line and the coordinates of the
# nearest point on the line.
#
# 1  Convert the line segment to a vector ('line_vec').
# 2  Create a vector connecting start to pnt ('pnt_vec').
# 3  Find the length of the line vector ('line_len').
# 4  Convert line_vec to a unit vector ('line_unitvec').
# 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
# 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
# 7  Ensure t is in the range 0 to 1.
# 8  Use t to get the nearest location on the line to the end
#    of vector pnt_vec_scaled ('nearest').
# 9  Calculate the distance from nearest to pnt_vec_scaled.
# 10 Translate nearest back to the start/end line.
# Malcolm Kesson 16 Dec 2012

def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)

def pnt2closestline(pnt, list_of_segments):
    '''Compute the closest distance between a point and each segment
    in list_of_segments [(start1,end1),(start2,end2),...]
    Return the idx of the closest segment, the distance to the point, and the
    point on the segment.
    '''
    dists = []
    nearests_on_line = []

    for segment in list_of_segments:
        res = pnt2line(pnt,segment[0],segment[1])
        dists.append(res[0])
        nearests_on_line.append(res[1])

    idx = np.argmin(dists)

    return idx, dists[idx], nearests_on_line[idx]



def dist_preservation_err(path_to_csv,with_plot=False,save_result=False):
    '''
    From a CSV file containing the VAE latent code of each single and the
    ground truth distance to initial state (ground truth measure of the phenotipycal
    change strengh), compute a score (MSE) based on how those distances are preserved.
    Indeed, we expect a good manifold to keep a smooth structure that depict the
    strengh of phenotype.
    The smaller the score is, the better it is.
    '''

    dimensionality = 3 #We infer dim is 3, control it later

    full_csv = pd.read_csv(path_to_csv)

    #Define where are the source phenotype in latent space
    red_cells = full_csv['GT_Shape']<0.15 #small shape factor are round red cells
    green_cells = full_csv['GT_Shape']>0.35

    #Take the 20 cells closest to the source phenotype, to define a green center and a red center
    x_reds = full_csv[red_cells].nsmallest(20,'GT_dist_toInit_state').x_coord.values
    y_reds = full_csv[red_cells].nsmallest(20,'GT_dist_toInit_state').y_coord.values
    x_greens = full_csv[green_cells].nsmallest(20,'GT_dist_toInit_state').x_coord.values
    y_greens = full_csv[green_cells].nsmallest(20,'GT_dist_toInit_state').y_coord.values
    try:
        z_reds = full_csv[red_cells].nsmallest(20,'GT_dist_toInit_state').z_coord.values
        z_greens = full_csv[green_cells].nsmallest(20,'GT_dist_toInit_state').z_coord.values
    except:
        print('2D latent space detected')
        dimensionality=2

    figplotly = plot_from_csv(path_to_csv,dim=dimensionality)

    ################################################
    # 2 DIMENSION ##########
    ################################################
    if dimensionality==2:
        red_latent_center = [np.mean(x_reds),np.mean(y_reds)]
        green_latent_center = [np.mean(x_greens),np.mean(y_greens)]

        #Define a center of for several different degree of phenotype strengh
        #Indeed, if we measure only distance betwen initial state and stronget phenotype,
        #it will favor completly straight manifold, which is not the aim.
        cluster_phenotype_centers = [] # 7 cluster x 4 midpoint x 2or3 coord
        cluster_list = np.unique(full_csv.GT_label.values)
        for cluster in cluster_list:
            intra_cluster = []
            for midpoint in [0.25,0.5,0.75]:
                #Consider the 15 single cell that are the closest to a phenotype strength midpoint
                cluster_index = full_csv['GT_label']==cluster
                sub_csv = full_csv[cluster_index]
                csv_sorted = sub_csv.iloc[(sub_csv['GT_dist_toInit_state']-midpoint).abs().argsort()[:15]]
                x_clus = np.mean(csv_sorted.x_coord.values)
                y_clus = np.mean(csv_sorted.y_coord.values)
                intra_cluster.append([x_clus,y_clus])

            cluster_index = full_csv['GT_label']==cluster
            x_clusters = full_csv[cluster_index].nlargest(15,'GT_dist_toInit_state').x_coord.values
            y_clusters = full_csv[cluster_index].nlargest(15,'GT_dist_toInit_state').y_coord.values
            intra_cluster.append([np.mean(x_clusters),np.mean(y_clusters)])

            cluster_phenotype_centers.append(intra_cluster)

        trace = go.Scatter(x=[red_latent_center[0],green_latent_center[0]],y=[red_latent_center[1],green_latent_center[1]],
            mode='markers',marker_symbol='x',marker_color='red',
            marker=dict(size=12, opacity=1),
            name=f'Centers')

        figplotly.add_traces(trace)

        cluster_phenotype_centers=np.asarray(cluster_phenotype_centers)
        trace2 = go.Scatter(x=np.squeeze(cluster_phenotype_centers[:6,:,0]),y=np.squeeze(cluster_phenotype_centers[:6,:,1]),
            mode='markers',marker_symbol='x',marker_color='black',
            marker=dict(size=8, opacity=1),
            name=f'Strong Phenotype')

        figplotly.add_traces(trace2)

        #Compute and add distance to maximum phenotype per cluster in GT dataframe
        distances = []
        Extremes = np.array([green_latent_center,red_latent_center])
        #Find distance to max Phenotype
        for index, row in full_csv.iterrows():
            #Find the GT strength of phenotype change of that cell
            GT_strength = row['GT_dist_toInit_state']
            GT_initial_state = row['GT_initial_state']
            init_state=Extremes[1]
            if GT_initial_state=='green':
                init_state=Extremes[0]

            GT_cluster = row['GT_label']
            #if (GT_strength < 0.25+0.05): #No midpoints are used
                #Dist between point and initial state
            #elif (GT_strength < 0.5 + 0.05): #Use one midpoints
                #Add segment initial state to midpoint 1 + segment midpoint 1 to point
            #elif (GT_strength < 0.75 + 0.05): #Use two midpoints
                #Add segment init-midpoint1 + midpoint1-midpoint2 + midpoint2-actualpoint
            #else: #Use the 3 midpoints
                #Add segment init-midpoint1 + midpoint1-midpoint2 + midpoint2-midpoint3 + midpoint3-actualpoint
            ind, dist = closest_point(np.array([row['x_coord'],row['y_coord']]),Extremes)
            distances.append(dist)
        full_csv['latent_dist_toInit_state'] = distances

        #Normalize to have the distance of center with strong_phenotype center = to 1
        cluster_list = np.unique(full_csv.GT_label.values)
        for i, cluster in enumerate(cluster_list):
            cluster_index = full_csv['GT_label']==cluster
            ind, normal_dist = closest_point(cluster_phenotype_centers[i,-1],Extremes)
            full_csv['latent_dist_toInit_state'][cluster_index] = full_csv['latent_dist_toInit_state'][cluster_index].values / normal_dist

    ################################################
    #% 3 DIMENSION ##########
    ################################################
    if dimensionality==3:
        red_latent_center = [np.mean(x_reds),np.mean(y_reds),np.mean(z_reds)]
        green_latent_center = [np.mean(x_greens),np.mean(y_greens),np.mean(z_greens)]


        #Define a center of for several different degree of phenotype strengh
        #Indeed, if we measure only distance betwen initial state and stronget phenotype,
        #it will favor completly straight manifold, which is not the aim.
        cluster_phenotype_midpoints = [] # 7 cluster x 3 midpoint x 2 green or red x 3dim
        cluster_max_centers = [] #7 x 3dim
        cluster_list = np.unique(full_csv.GT_label.values)
        for cluster in cluster_list:
            intra_cluster = []
            for midpoint in [0.25,0.5,0.75]:
                #Consider the 15 single cell that are the closest to a phenotype strength midpoint
                cluster_index = full_csv['GT_label']==cluster
                green_index = full_csv['GT_initial_state']=='green'
                red_index = full_csv['GT_initial_state']=='red'
                sub_csv_green = full_csv[(cluster_index) & (green_index)]
                sub_csv_red = full_csv[(cluster_index) & (red_index)]
                csv_sorted_green = sub_csv_green.iloc[(sub_csv_green['GT_dist_toInit_state']-midpoint).abs().argsort()[:15]]
                csv_sorted_red = sub_csv_red.iloc[(sub_csv_red['GT_dist_toInit_state']-midpoint).abs().argsort()[:15]]

                x_clus_green = np.mean(csv_sorted_green.x_coord.values)
                y_clus_green = np.mean(csv_sorted_green.y_coord.values)
                z_clus_green = np.mean(csv_sorted_green.z_coord.values)

                x_clus_red = np.mean(csv_sorted_red.x_coord.values)
                y_clus_red = np.mean(csv_sorted_red.y_coord.values)
                z_clus_red = np.mean(csv_sorted_red.z_coord.values)
                intra_cluster.append([[x_clus_green,y_clus_green,z_clus_green],[x_clus_red,y_clus_red,z_clus_red]])

            cluster_index = full_csv['GT_label']==cluster
            x_clusters = full_csv[cluster_index].nlargest(15,'GT_dist_toInit_state').x_coord.values
            y_clusters = full_csv[cluster_index].nlargest(15,'GT_dist_toInit_state').y_coord.values
            z_clusters = full_csv[cluster_index].nlargest(15,'GT_dist_toInit_state').z_coord.values
            cluster_max_centers.append([np.mean(x_clusters),np.mean(y_clusters),np.mean(z_clusters)])

            cluster_phenotype_midpoints.append(intra_cluster)

        cluster_phenotype_midpoints = np.array(cluster_phenotype_midpoints)
        cluster_max_centers = np.array(cluster_max_centers)

        #Plot the manifold backbone :
        # Initial state to midpoints to strongest phenotype

        cluster_list = np.unique(full_csv[full_csv['GT_label']!=7].GT_label.values)
        Extremes = np.array([green_latent_center,red_latent_center])
        backbone = []
        traces = []
        for cluster in cluster_list:
            temp = []
            temp_g = [Extremes[0],cluster_phenotype_midpoints[cluster-1,0,0,:],cluster_phenotype_midpoints[cluster-1,1,0,:],cluster_phenotype_midpoints[cluster-1,2,0,:],cluster_max_centers[cluster-1,:]]
            temp_r = [cluster_phenotype_midpoints[cluster-1,2,1,:],cluster_phenotype_midpoints[cluster-1,1,1,:],cluster_phenotype_midpoints[cluster-1,0,1,:],Extremes[1]]
            backbone_c = np.array(temp_g+temp_r)
            backbone.append(backbone_c)

            scatter = go.Scatter3d(x=backbone_c[:,0],y=backbone_c[:,1],z=backbone_c[:,2],
                mode='lines+markers',marker_symbol='x',marker=dict(size=10, opacity=1),
                name=f'backbone cluster {cluster}', marker_color=plotly.colors.qualitative.Plotly[cluster-1],
                line_width=6)
            traces.append(scatter)

        figplotly.add_traces(traces)
        # trace = go.Scatter3d(x=[red_latent_center[0],green_latent_center[0]],y=[red_latent_center[1],green_latent_center[1]],
        #     z=[red_latent_center[2],green_latent_center[2]],mode='markers',marker_symbol='x',marker_color='red',
        #     marker=dict(size=12, opacity=1),
        #     name=f'Centers')
        #
        # figplotly.add_trace(trace)
        #
        # cluster_phenotype_midpoints=np.asarray(cluster_phenotype_midpoints)
        # trace2 = go.Scatter3d(x=cluster_phenotype_midpoints[:,:,:,0].flatten(),y=cluster_phenotype_midpoints[:,:,:,1].flatten(),
        #     z=cluster_phenotype_midpoints[:,:,:,2].flatten(),mode='markers',marker_symbol='x',marker_color='black',
        #     marker=dict(size=8, opacity=1),
        #     name=f'Strong Phenotype')
        # figplotly.add_trace(trace2)
        #
        # cluster_max_centers=np.asarray(cluster_max_centers)
        # trace3 = go.Scatter3d(x=cluster_max_centers[:,0].flatten(),y=cluster_max_centers[:,1].flatten(),
        #     z=cluster_max_centers[:,2].flatten(),mode='markers',marker_symbol='x',marker_color='black',
        #     marker=dict(size=8, opacity=1),
        #     name=f'Strong Phenotype')
        # figplotly.add_trace(trace3)

        #Compute and add distance to initial state per cluster in GT dataframe
        #Measure it as segment throughout midpoints to not favor straight manifold
        #Find always closest point on segment to not favor narrow manifold
        distances = []

        #Find distance to max Phenotype
        for index, row in full_csv.iterrows():
            #Find the GT strength of phenotype change of that cell
            GT_strength = row['GT_dist_toInit_state']
            GT_initial_state = row['GT_initial_state']
            ind_init = 1
            if GT_initial_state=='green':
                ind_init = 0
            GT_cluster = row['GT_label']

            seg1 = (Extremes[ind_init],cluster_phenotype_midpoints[GT_cluster-1,0,ind_init,:])
            seg2 = (cluster_phenotype_midpoints[GT_cluster-1,0,ind_init,:],cluster_phenotype_midpoints[GT_cluster-1,1,ind_init,:])
            seg3 = (cluster_phenotype_midpoints[GT_cluster-1,1,ind_init,:],cluster_phenotype_midpoints[GT_cluster-1,2,ind_init,:])
            seg4 = (cluster_phenotype_midpoints[GT_cluster-1,2,ind_init,:],cluster_max_centers[GT_cluster-1,:])

            list_of_seg = [(seg1[0],seg1[1]),(seg2[0],seg2[1]),(seg3[0],seg3[1]),(seg4[0],seg4[1])]
            actual_point = np.array([row['x_coord'],row['y_coord'],row['z_coord']])
            #id of clost segment, dist to the segment, coord of closest point on line
            idx, dist_to_seg, nearests_on_line = pnt2closestline(actual_point,list_of_seg)
            nearests_on_line = np.round(np.array(nearests_on_line),4)
            dist = 0
            if (idx == 0): #No midpoints are used and use dist return
                if (np.all(np.round(nearests_on_line,4)==np.round(seg1[0],4))) :
                    dist = dist_to_seg
                else:
                    dist = sqr_distance(seg1[0],np.array(nearests_on_line))

            elif (idx==1): #Use first midpoint
                d1 = sqr_distance(seg1[0],seg1[1])
                d2 = sqr_distance(seg2[0],np.array(nearests_on_line))
                dist = d1+d2

            elif (idx==2): #Use two midpoints
                d1 = sqr_distance(seg1[0],seg1[1])
                d2 = sqr_distance(seg2[0],seg2[1])
                d3 = sqr_distance(seg3[0],np.array(nearests_on_line))
                dist = d1+d2+d3

            elif (idx == 3): #Use all 3 midpoints
                if not(np.all(np.round(nearests_on_line,4)==np.round(seg4[1],4))):
                    d1 = sqr_distance(seg1[0],seg1[1])
                    d2 = sqr_distance(seg2[0],seg2[1])
                    d3 = sqr_distance(seg3[0],seg3[1])
                    d4 = sqr_distance(seg4[0],np.array(nearests_on_line))
                    dist = d1+d2+d3+d4
                else:
                    d1 = sqr_distance(seg1[0],seg1[1])
                    d2 = sqr_distance(seg2[0],seg2[1])
                    d3 = sqr_distance(seg3[0],seg3[1])
                    d4 = sqr_distance(seg4[0],seg4[1])
                    dist = d1+d2+d3+d4+dist_to_seg



            # if (GT_strength < 0.25+0.05): #No midpoints are used
            #     #Dist between point and initial state
            #     dist = sqr_distance(np.array([row['x_coord'],row['y_coord'],row['z_coord']]),Extremes[ind_init])
            # elif (GT_strength < 0.5 + 0.05): #Use one midpoints
            #     #Add segment initial state to midpoint 1 + segment midpoint 1 to point
            #     d1 = sqr_distance(Extremes[ind_init],cluster_phenotype_midpoints[GT_cluster-1,0,ind_init,:])
            #     d2 = sqr_distance(cluster_phenotype_midpoints[GT_cluster-1,0,ind_init,:],np.array([row['x_coord'],row['y_coord'],row['z_coord']]))
            #     dist = d1+d2
            #
            # elif (GT_strength < 0.75 + 0.05): #Use two midpoints
            #     #Add segment init-midpoint1 + midpoint1-midpoint2 + midpoint2-actualpoint
            #     d1 = sqr_distance(Extremes[ind_init],cluster_phenotype_midpoints[GT_cluster-1,0,ind_init,:])
            #     d2 = sqr_distance(cluster_phenotype_midpoints[GT_cluster-1,0,ind_init,:],cluster_phenotype_midpoints[GT_cluster-1,1,ind_init,:])
            #     d3 = sqr_distance(cluster_phenotype_midpoints[GT_cluster-1,1,ind_init,:],np.array([row['x_coord'],row['y_coord'],row['z_coord']]))
            #     dist = d1+d2+d3
            # else: #Use the 3 midpoints
            #     #Add segment init-midpoint1 + midpoint1-midpoint2 + midpoint2-midpoint3 + midpoint3-actualpoint
            #     d1 = sqr_distance(Extremes[ind_init],cluster_phenotype_midpoints[GT_cluster-1,0,ind_init,:])
            #     d2 = sqr_distance(cluster_phenotype_midpoints[GT_cluster-1,0,ind_init,:],cluster_phenotype_midpoints[GT_cluster-1,1,ind_init,:])
            #     d3 = sqr_distance(cluster_phenotype_midpoints[GT_cluster-1,1,ind_init,:],cluster_phenotype_midpoints[GT_cluster-1,2,ind_init,:])
            #     d4 = sqr_distance(cluster_phenotype_midpoints[GT_cluster-1,2,ind_init,:],np.array([row['x_coord'],row['y_coord'],row['z_coord']]))
            #     dist = d1+d2+d3+d4

            distances.append(dist)
        full_csv['latent_dist_toInit_state'] = distances

        #Normalize to have the distance of center with strong_phenotype center = to 1
        #Take the path throughout all the midpoints
        cluster_list = np.unique(full_csv.GT_label.values)
        for i, cluster in enumerate(cluster_list):
            for j, init_state in enumerate(['green','red']): #distance towards green or towards red
                cluster_index = full_csv['GT_label']==cluster
                init_state_index = full_csv['GT_initial_state']==init_state

                d1 = sqr_distance(Extremes[j],cluster_phenotype_midpoints[cluster-1,0,j,:])
                d2 = sqr_distance(cluster_phenotype_midpoints[cluster-1,0,j,:],cluster_phenotype_midpoints[cluster-1,1,j,:])
                d3 = sqr_distance(cluster_phenotype_midpoints[cluster-1,1,j,:],cluster_phenotype_midpoints[cluster-1,2,j,:])
                d4 = sqr_distance(cluster_phenotype_midpoints[cluster-1,2,j,:],cluster_max_centers[cluster-1,:])

                normal_dist = d1+d2+d3+d4
                full_csv['latent_dist_toInit_state'][cluster_index & init_state_index] = full_csv['latent_dist_toInit_state'][cluster_index & init_state_index].values / normal_dist
    ################################################
    # Save and visual assement  ##################
    ################################################
    if save_result:
        full_csv.to_csv(path_to_csv,index=False)


    ## Plot the ordered results for visual assessment
    full_csv = full_csv.sort_values(by='GT_dist_toInit_state')

    #Disregard cluster 7 MSE because no coherent manifold
    no_cluster_7 = full_csv['GT_label']!=7

    line1 = go.Scatter(y=full_csv[no_cluster_7].GT_dist_toInit_state.values,
        mode='lines',name='GT_distance',line=dict(width=4))
    line2 = go.Scatter(y=full_csv[no_cluster_7].latent_dist_toInit_state.values,
        mode='markers',name='Latent_distance',marker=dict(size=3))

    #Moving average on distance on latent space to assess monotony
    dist_dataframe = full_csv[no_cluster_7]
    dist_dataframe.reset_index()
    stride = 50
    dist_dataframe['SMA_10']=np.nan
    dist_dataframe.loc[::stride,'SMA_10'] = dist_dataframe.loc[:,'latent_dist_toInit_state'].rolling(window=150).mean()
    #dist_dataframe['SMA_10'] = dist_dataframe.loc[:,'latent_dist_toInit_state'].rolling(window=50).mean()
    print(dist_dataframe.SMA_10)
    #dist_dataframe['SMA_10'].fillna(0)

    line3 = go.Scatter(y=dist_dataframe['SMA_10'].values,
        mode='lines',connectgaps=True,name='Rolling avg',line=dict(width=2))



    layout=go.Layout(title='test')
    fig_te = go.Figure(data=[line1,line2,line3],layout=layout)

    ################################################
    # Numerical value assessment -- MSE ##########
    ################################################
    # Calculate a score (MSE) for the distance_to_strong_phenotype fitting
    #Disregard cluster 7 MSE because no coherent manifold
    no_cluster_7 = full_csv['GT_label']!=7
    GT_distance = full_csv[no_cluster_7].GT_dist_toInit_state.values
    latent_distance = full_csv[no_cluster_7].latent_dist_toInit_state.values

    overall_mse = metrics.mean_squared_error(GT_distance, latent_distance)
    spearman_r = stats.spearmanr(GT_distance,latent_distance)
    kendall_r = stats.kendalltau(GT_distance,latent_distance)

    mse_per_cluster = []
    cluster_list = np.unique(full_csv.GT_label.values)
    for cluster in cluster_list:
        cluster_index = full_csv['GT_label']==cluster
        mse_per_cluster.append(metrics.mean_squared_error(full_csv[cluster_index].GT_dist_toInit_state.values,full_csv[cluster_index].latent_dist_toInit_state.values))

    if with_plot:

        title = f'VAE, dist to initial state metric | Spearman Coeff : {spearman_r[0]:.4f}, Kendall Coeff : {kendall_r[4]:.3f}'
        fig_te.update_layout(margin=dict(l=1.1,r=1.1,b=1.1,t=30),showlegend=True,legend=dict(y=-.1),title=dict(text=title))
        fig_te.update_layout(title={'yref':'paper','y':1,'yanchor':'bottom'},title_x=0.5)
        fig_te.show()

        fig2 = px.bar(x=cluster_list[:6],y=mse_per_cluster[:6])
        fig2.update_layout(title='MSE per cluster')
        fig2.show()

    return overall_mse, spearman_r, kendall_r, figplotly


def MINE_metric(path_to_csv):

    batch_size = 250
    input_size = 64
    _, infer_dataloader = get_inference_dataset('DataSets/Synthetic_Data_1',batch_size,input_size,shuffle=True,droplast=True)

    MINEnet = MINE(input_size*input_size*3,zdim=3)
    MINEnet.cuda()

    history_MI = train_MINE(MINEnet,path_to_csv,100,infer_dataloader,train_GPU=True)
    fig, ax = plt.subplots(1,1)
    ax.plot(history_MI)

    return history_MI[-1]


#Load the appropriate CSV file
name_of_csv1 = 'DataSets/Sacha_Metadata_3dlatentVAE_20200523.csv'
name_of_csv2 = 'DataSets/Sacha_Metadata_3dlatentVAEFAIL_20200524.csv'
name_of_csv3 = 'DataSets/Sacha_Metadata_3dlatentVAEbigFAIL_20200525.csv'

#compare_models([name_of_csv1,name_of_csv2,name_of_csv3],Metrics=[False,False,True],num_iteration=10)

mse,spearman_r, kendall_r, figplotly = dist_preservation_err(name_of_csv3,with_plot=True,save_result=False)
plotly.offline.plot(figplotly, filename='backboneModel-3.html', auto_open=False)

#%%
print(spearman_r)
print(kendall_r)
#MI = MINE_metric(name_of_csv1)




plotly.colors.qualitative.Plotly[0]
data = pd.read_csv(name_of_csv3)
data['GT_label'] = data['GT_label'].astype(str)
fig = px.scatter_3d(data,x='x_coord',y='y_coord',z='z_coord',color='GT_colorB')
plotly.offline.plot(fig, filename='GT_blue_model3.html', auto_open=True)
