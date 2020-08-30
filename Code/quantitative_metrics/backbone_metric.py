# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-07-03T09:35:12+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-07-06T10:44:57+10:00


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import metrics
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline
from helpers import plot_from_csv
from scipy import stats
import pickle as pkl


############################################################
############################################################
### Backbone generation and distance to initial state score
############################################################
############################################################

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



def dist_preservation_err(path_to_csv,low_dim_names=['x_coord','y_coord','z_coord'],overwrite_csv=False,save_path=None):
    '''
    From a CSV file containing the VAE latent code of each single cell and the
    ground truth distance to initial state (ground truth measure of the phenotipycal
    change strengh), compute a score (rank correlation) based on how the rank are preserved.
    Indeed, we expect a good manifold to keep a smooth structure that depict the
    strengh of phenotype.
    The closest to 1 the score is, the better it is

    path_to_csv can be a string (path to csv file) or a DataFrame

    NOTE : The csv file must countain ground turth information from BBBC (GT_shape, GT_dist_toInit_state)
    '''

    dimensionality = 3 #We infer dim is 3, control it later

    if isinstance(path_to_csv,str):
        full_csv = pd.read_csv(path_to_csv)
    else :
        full_csv = path_to_csv

    #Define where are the source phenotype in latent space
    red_cells = full_csv['GT_Shape']<0.15 #small shape factor are round red cells
    green_cells = full_csv['GT_Shape']>0.35

    #Take the 20 cells closest to the source phenotype, to define a green center and a red center
    x_reds = full_csv[red_cells].nsmallest(20,'GT_dist_toInit_state')[low_dim_names[0]].values
    y_reds = full_csv[red_cells].nsmallest(20,'GT_dist_toInit_state')[low_dim_names[1]].values
    x_greens = full_csv[green_cells].nsmallest(20,'GT_dist_toInit_state')[low_dim_names[0]].values
    y_greens = full_csv[green_cells].nsmallest(20,'GT_dist_toInit_state')[low_dim_names[1]].values
    try:
        z_reds = full_csv[red_cells].nsmallest(20,'GT_dist_toInit_state')[low_dim_names[2]].values
        z_greens = full_csv[green_cells].nsmallest(20,'GT_dist_toInit_state')[low_dim_names[2]].values
    except:
        print('2D latent space detected')
        dimensionality=2

    figplotly = plot_from_csv(path_to_csv,low_dim_names,dim=dimensionality)

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

                x_clus_green = np.mean(csv_sorted_green[low_dim_names[0]].values)
                y_clus_green = np.mean(csv_sorted_green[low_dim_names[1]].values)
                z_clus_green = np.mean(csv_sorted_green[low_dim_names[2]].values)

                x_clus_red = np.mean(csv_sorted_red[low_dim_names[0]].values)
                y_clus_red = np.mean(csv_sorted_red[low_dim_names[1]].values)
                z_clus_red = np.mean(csv_sorted_red[low_dim_names[2]].values)
                intra_cluster.append([[x_clus_green,y_clus_green,z_clus_green],[x_clus_red,y_clus_red,z_clus_red]])

            cluster_index = full_csv['GT_label']==cluster
            x_clusters = full_csv[cluster_index].nlargest(15,'GT_dist_toInit_state')[low_dim_names[0]].values
            y_clusters = full_csv[cluster_index].nlargest(15,'GT_dist_toInit_state')[low_dim_names[1]].values
            z_clusters = full_csv[cluster_index].nlargest(15,'GT_dist_toInit_state')[low_dim_names[2]].values
            cluster_max_centers.append([np.mean(x_clusters),np.mean(y_clusters),np.mean(z_clusters)])

            cluster_phenotype_midpoints.append(intra_cluster)

        cluster_phenotype_midpoints = np.array(cluster_phenotype_midpoints)
        cluster_max_centers = np.array(cluster_max_centers)

        #Plot the manifold backbone :
        # Initial state to midpoints to strongest phenotype

        cluster_list = np.unique(full_csv[full_csv['GT_label']!=7].GT_label.values)
        Extremes = np.array([green_latent_center,red_latent_center])
        backbone = [] # 6 x 9 x 3     6 backbone, 9 point 8 segment, 3 coordinates
        traces = []
        for cluster in cluster_list:
            cluster = int(cluster)
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

        distances = []

        #Find distance to max Phenotype
        for index, row in full_csv.iterrows():
            #Find the GT strength of phenotype change of that cell
            GT_strength = row['GT_dist_toInit_state']
            GT_initial_state = row['GT_initial_state']
            ind_init = 1
            if GT_initial_state=='green':
                ind_init = 0
            GT_cluster = int(row['GT_label'])

            seg1 = (Extremes[ind_init],cluster_phenotype_midpoints[GT_cluster-1,0,ind_init,:])
            seg2 = (cluster_phenotype_midpoints[GT_cluster-1,0,ind_init,:],cluster_phenotype_midpoints[GT_cluster-1,1,ind_init,:])
            seg3 = (cluster_phenotype_midpoints[GT_cluster-1,1,ind_init,:],cluster_phenotype_midpoints[GT_cluster-1,2,ind_init,:])
            seg4 = (cluster_phenotype_midpoints[GT_cluster-1,2,ind_init,:],cluster_max_centers[GT_cluster-1,:])

            list_of_seg = [(seg1[0],seg1[1]),(seg2[0],seg2[1]),(seg3[0],seg3[1]),(seg4[0],seg4[1])]
            actual_point = np.array([row[low_dim_names[0]],row[low_dim_names[1]],row[low_dim_names[2]]])
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

            distances.append(dist)
        full_csv['latent_dist_toInit_state'] = distances

        #Normalize to have the distance of center with strong_phenotype center = to 1
        #Take the path throughout all the midpoints
        cluster_list = np.unique(full_csv.GT_label.values)
        for i, cluster in enumerate(cluster_list):
            cluster = int(cluster)
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
    if overwrite_csv: #Add new column to the csv
        full_csv.to_csv(path_to_csv,index=False)


    ## Plot the ordered results for visual assessment
    full_csv = full_csv.sort_values(by='GT_dist_toInit_state')

    #Disregard cluster 7 because no coherent manifold
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

    line3 = go.Scatter(y=dist_dataframe['SMA_10'].values,
        mode='lines',connectgaps=True,name='Rolling avg',line=dict(width=2))

    layout=go.Layout(title='test')
    fig_te = go.Figure(data=[line1,line2,line3],layout=layout)

    ################################################
    # Numerical value assessment -- Rank Correlation ##########
    ################################################
    # Calculate a score for the distance_to_strong_phenotype fitting
    #Disregard cluster 7 because no coherent manifold
    no_cluster_7 = full_csv['GT_label']!=7
    GT_distance = full_csv[no_cluster_7].GT_dist_toInit_state.values
    latent_distance = full_csv[no_cluster_7].latent_dist_toInit_state.values

    spearman_r = stats.spearmanr(GT_distance,latent_distance)
    kendall_r = stats.kendalltau(GT_distance,latent_distance)

    spearman_per_cluster = []
    cluster_list = np.unique(full_csv.GT_label.values)
    for cluster in cluster_list:
        cluster_index = full_csv['GT_label']==cluster
        spearman_per_cluster.append(stats.spearmanr(full_csv[cluster_index].GT_dist_toInit_state.values,full_csv[cluster_index].latent_dist_toInit_state.values)[0])

    title = f'VAE, dist to initial state metric | Spearman Coeff : {spearman_r[0]:.4f}, Kendall Coeff : {kendall_r[0]:.3f}'
    fig_te.update_layout(margin=dict(l=1.1,r=1.1,b=1.1,t=30),showlegend=True,legend=dict(y=-.1),title=dict(text=title))
    fig_te.update_layout(title={'yref':'paper','y':1,'yanchor':'bottom'},title_x=0.5)
    #fig_te.show()

    fig2 = px.bar(x=cluster_list[:6],y=spearman_per_cluster[:6])
    fig2.update_layout(title='Spearman coeff per cluster')
    #fig2.show()

    if save_path != None:
        plotly.offline.plot(figplotly, filename=f'{save_path}/backbone_plot.html', auto_open=False)
        plotly.offline.plot(fig_te, filename=f'{save_path}/correlation_fit.html', auto_open=False)
        backbone_pkl = save_path+'/backbone_points.pkl'
        with open(backbone_pkl, 'wb') as f:
            pkl.dump(backbone, f, protocol=pkl.HIGHEST_PROTOCOL)
        correlation_score_df = pd.DataFrame({'spearman_r':spearman_r[0],'kendall_r':kendall_r[0]},index=[0])
        correlation_score_df.to_csv(f'{save_path}/correlation_fit.csv')

    return backbone, spearman_r[0], kendall_r[0]
