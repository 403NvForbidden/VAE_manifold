import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVC
import pandas as pd
import plotly



### prepare data 

df = pd.read_csv('/mnt/Linux_Storage/outputs/vaDE/embeded_data.csv')
non_last_cluster = df['GT_label'] != 7
strong_phenotype = df['GT_dist_toInit_state'] >= 0.5
df = df[non_last_cluster & strong_phenotype].reset_index(drop=False, inplace=False)
# For NN, the class must be between 0 - num_class-1
df['GT_label'] = df['GT_label'].subtract(1)

# Built metacluster (1&2 vs 3&4 vs 5&6)

df['GT_label'].replace(1, 0, inplace=True)
df['GT_label'].replace(2, 1, inplace=True)
df['GT_label'].replace(3, 1, inplace=True)
df['GT_label'].replace(4, 2, inplace=True)
df['GT_label'].replace(5, 2, inplace=True)


mesh_size = .02
margin = 0

X = df[['z0', 'z1', 'z2']]
y = df['GT_label']

# Condition the model on sepal width and length, predict the petal width
svc = SVC(kernel='linear')
svc.fit(X,y)

z = lambda x,y: (-svc.intercept_[0]-svc.coef_[0][0]*x-svc.coef_[0][1]*y) / svc.coef_[0][2]
qqq = z(X,y)

# Create a mesh grid on which we will run our model
x_min, x_max = X.z0.min() - margin, X.z0.max() + margin
y_min, y_max = X.z1.min() - margin, X.z1.max() + margin
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)

# Generate the plot
fig = px.scatter_3d(df, x='z0', y='z1', z='z2', opacity=1,color='GT_label')
fig.update_traces(marker=dict(size=3))
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=True, legend=dict(y=-.1))
fig.add_traces(go.Surface(x=xx, y=yy, z=z(xx,yy),name='surface', opacity=0.5,showscale=False))
plotly.offline.plot(fig, filename='/mnt/Linux_Storage/outputs/vaDE/SVC_1.html', auto_open=True)