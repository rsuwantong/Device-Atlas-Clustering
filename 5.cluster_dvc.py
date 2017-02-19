# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:37:43 2017

@author: T900930
"""

###############################################################################
# Name: Mobile Atlas Data Pre-processing
# Version:
#   2017/02/09 RS: Initial version
###############################################################################


# -*- coding: utf-8 -*-
"""
Contents


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import os

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering

os.chdir('D:/Tapad_UC1/Mobile_Atlas/Device_clustering')
df = pd.read_csv('techname_matched2atlas.csv',sep='\s*,\s*', header = 0, encoding='utf-8')
df.head(20)
print(df.columns)
df.shape #(9100,12)

#======= 1. Data cleansing ======

df['AGE'] = 2017+2/12-df['TIME_RELEASED']



fig_size = plt.rcParams["figure.figsize"]
print(fig_size) 
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.DIAGONAL_SCREEN_SIZE)
threedee.set_xlabel('P (eu)')
threedee.set_ylabel('A (year)')
threedee.set_zlabel('S (inch)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.DIAGONAL_SCREEN_SIZE.min(),20])
threedee.view_init(15,15)
plt.show()



fig_size = plt.rcParams["figure.figsize"]
print(fig_size) 
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS)
threedee.set_xlabel('P (eu)')
threedee.set_ylabel('A (year)')
threedee.set_zlabel('Camera (pixel)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()



# We will take off the ouliers: customers with very high monetary and recency (99% quantile will be used).

df = df[(df.PRICE_RELEASED <= 1000) & (df.AGE <= 6) & (df.CAMERA_PIXELS <= 25)
         & (df.DIAGONAL_SCREEN_SIZE <= 10)]
df.shape # (3461,16) # 

fig_size = plt.rcParams["figure.figsize"]
print(fig_size) 
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS)
threedee.set_xlabel('P (eu)')
threedee.set_ylabel('A (year)')
threedee.set_zlabel('Camera (pixel)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()


#======= 2. Clustering using RFM  ======


X=df[['PRICE_RELEASED','AGE','CAMERA_PIXELS']]
X = X.set_index(df['MODEL'])

scaler = StandardScaler()
# Use standard scale to make mean = 0 and std = 1
XScale = scaler.fit_transform(X)

# 2.1 Kmeans clustering

# We use the elbow and silhouette methods to visualize the optimal number of clusters
distortion_km = []
silhouette_km = []
for i in range(2, 21):
   km = KMeans(n_clusters = i,
               n_init = 50,
               max_iter = 1000,
               random_state = 0)
   km.fit(XScale)
   distortion_km.append(km.inertia_)
   silhouette_km.append(metrics.silhouette_score(XScale, km.labels_))

   
plt.plot(range(2,21), distortion_km, marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
# The elbow method suggests the optimal number of clusters around 3-5

plt.plot(range(2,21), silhouette_km, marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette')
plt.show()
# The silouhette method is not conclusive.
# We will therefore use 3, 4 and 5 as the number of clusters to be tested.

kmeans7 = KMeans(n_clusters=7, random_state=0).fit(XScale)
kmeans7.labels_
df['Cluster_km7'] = kmeans7.labels_

kmeans8 = KMeans(n_clusters=8, random_state=0).fit(XScale)
kmeans8.labels_
df['Cluster_km8'] = kmeans8.labels_

kmeans10 = KMeans(n_clusters=10, random_state=0).fit(XScale)
kmeans10.labels_
df['Cluster_km10'] = kmeans10.labels_


kmeans12 = KMeans(n_clusters=12, random_state=0).fit(XScale)
kmeans12.labels_
df['Cluster_km12'] = kmeans12.labels_

# 2.2 DBScan

# We try to visualize a good value of epsilon for dbscan parameterization 
# by computing the distance between nearest neighbour of the data
nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

distances =pd.DataFrame( {'distances': distances[:,1],
                              })
distances = distances.sort_values('distances',ascending=True)
distances = distances.reset_index(drop=True)

plt.plot(range(0, len(distances)),distances, marker = 'o')
plt.xlabel('Object')
plt.ylabel('k distance')
plt.show()
# The epsilon value around 20-30 should represent a distance between clusters

db = DBSCAN(eps=2, min_samples=10,metric='euclidean').fit(XScale)
df['Cluster_db'] = db.labels_
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
df.Cluster_db.unique()
# We get only 1 cluster from the dbscan method which may due to the fact that all the points are closely packed together.

# 2.3 Hierarchical agglomerative clustering with connectivity constraints

# we choose the ward linkage criteria for merging strategy (minimizing the sum of squared differences within all clusters)


connectivity = kneighbors_graph(XScale, n_neighbors=10, include_self=False)
ward = AgglomerativeClustering(n_clusters=7, connectivity=connectivity,
                               linkage='ward').fit(XScale)
df['Cluster_hclust7'] = ward.labels_

ward = AgglomerativeClustering(n_clusters=8, connectivity=connectivity,
                               linkage='ward').fit(XScale)
df['Cluster_hclust8'] = ward.labels_

ward = AgglomerativeClustering(n_clusters=10, connectivity=connectivity,
                               linkage='ward').fit(XScale)
df['Cluster_hclust10'] = ward.labels_


ward = AgglomerativeClustering(n_clusters=12, connectivity=connectivity,
                               linkage='ward').fit(XScale)
df['Cluster_hclust12'] = ward.labels_


# 2.3 2.4 Clustering result visualization

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.Cluster_km7)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()



threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.Cluster_km8)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.Cluster_km10)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.Cluster_km12)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

# kmean and hclust give similar clustering results

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.Cluster_hclust7)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()


threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.Cluster_hclust8)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.Cluster_hclust10)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.Cluster_hclust12)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()


# With screen size

X=df[['PRICE_RELEASED','AGE','CAMERA_PIXELS', 'DIAGONAL_SCREEN_SIZE']]
X = X.set_index(df['MODEL'])

scaler = StandardScaler()
# Use standard scale to make mean = 0 and std = 1
XScale = scaler.fit_transform(X)

# 2.1 Kmeans clustering

# We use the elbow and silhouette methods to visualize the optimal number of clusters
distortion_km = []
silhouette_km = []
for i in range(2, 21):
   km = KMeans(n_clusters = i,
               n_init = 50,
               max_iter = 1000,
               random_state = 0)
   km.fit(XScale)
   distortion_km.append(km.inertia_)
   silhouette_km.append(metrics.silhouette_score(XScale, km.labels_))

   
plt.plot(range(2,21), distortion_km, marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
# The elbow method suggests the optimal number of clusters around 7

plt.plot(range(2,21), silhouette_km, marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette')
plt.show()
# The silouhette method proposed 3 (too small...)

kmeans7 = KMeans(n_clusters=7, random_state=0).fit(XScale)
kmeans7.labels_
df['ClusterScrCam_km7'] = kmeans7.labels_

kmeans8 = KMeans(n_clusters=8, random_state=0).fit(XScale)
kmeans8.labels_
df['ClusterScrCam_km8'] = kmeans8.labels_

kmeans10 = KMeans(n_clusters=10, random_state=0).fit(XScale)
kmeans10.labels_
df['ClusterScrCam_km10'] = kmeans10.labels_


kmeans12 = KMeans(n_clusters=12, random_state=0).fit(XScale)
kmeans12.labels_
df['ClusterScrCam_km12'] = kmeans12.labels_

# 2.2 DBScan

# We try to visualize a good value of epsilon for dbscan parameterization 
# by computing the distance between nearest neighbour of the data
nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

distances =pd.DataFrame( {'distances': distances[:,1],
                              })
distances = distances.sort_values('distances',ascending=True)
distances = distances.reset_index(drop=True)

plt.plot(range(0, len(distances)),distances, marker = 'o')
plt.xlabel('Object')
plt.ylabel('k distance')
plt.show()
# The epsilon value around 20-30 should represent a distance between clusters

db = DBSCAN(eps=2, min_samples=10,metric='euclidean').fit(XScale)
df['Cluster_db'] = db.labels_
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
df.Cluster_db.unique()
# We get only 1 cluster from the dbscan method which may due to the fact that all the points are closely packed together.

# 2.3 Hierarchical agglomerative clustering with connectivity constraints

# we choose the ward linkage criteria for merging strategy (minimizing the sum of squared differences within all clusters)


connectivity = kneighbors_graph(XScale, n_neighbors=10, include_self=False)
ward = AgglomerativeClustering(n_clusters=7, connectivity=connectivity,
                               linkage='ward').fit(XScale)
df['ClusterScrCam_hclust7'] = ward.labels_

ward = AgglomerativeClustering(n_clusters=8, connectivity=connectivity,
                               linkage='ward').fit(XScale)
df['ClusterScrCam_hclust8'] = ward.labels_

ward = AgglomerativeClustering(n_clusters=10, connectivity=connectivity,
                               linkage='ward').fit(XScale)
df['ClusterScrCam_hclust10'] = ward.labels_


ward = AgglomerativeClustering(n_clusters=12, connectivity=connectivity,
                               linkage='ward').fit(XScale)
df['ClusterScrCam_hclust12'] = ward.labels_


# 2.3 2.4 Clustering result visualization

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.ClusterScrCam_km7)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.DIAGONAL_SCREEN_SIZE, c= df.ClusterScrCam_km7)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('SCREEN (inch)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.DIAGONAL_SCREEN_SIZE.min(),df.DIAGONAL_SCREEN_SIZE.max()])
threedee.view_init(15,15)
plt.show()




threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.ClusterScrCam_km8)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.ClusterScrCam_km10)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.ClusterScrCam_km12)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

# kmean and hclust give similar clustering results

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.ClusterScrCam_hclust7)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()


threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.ClusterScrCam_hclust8)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.ClusterScrCam_hclust10)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.ClusterScrCam_hclust12)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()


# Using only screen size

X=df[['PRICE_RELEASED','AGE', 'DIAGONAL_SCREEN_SIZE']]
X = X.set_index(df['MODEL'])

scaler = StandardScaler()
# Use standard scale to make mean = 0 and std = 1
XScale = scaler.fit_transform(X)

# 2.1 Kmeans clustering

# We use the elbow and silhouette methods to visualize the optimal number of clusters
distortion_km = []
silhouette_km = []
for i in range(2, 21):
   km = KMeans(n_clusters = i,
               n_init = 50,
               max_iter = 1000,
               random_state = 0)
   km.fit(XScale)
   distortion_km.append(km.inertia_)
   silhouette_km.append(metrics.silhouette_score(XScale, km.labels_))

   
plt.plot(range(2,21), distortion_km, marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
# The elbow method suggests the optimal number of clusters around 3-5

plt.plot(range(2,21), silhouette_km, marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette')
plt.show()
# The silouhette method is not conclusive.
# We will therefore use 3, 4 and 5 as the number of clusters to be tested.

kmeans7 = KMeans(n_clusters=7, random_state=0).fit(XScale)
kmeans7.labels_
df['ClusterScr_km7'] = kmeans7.labels_

kmeans8 = KMeans(n_clusters=8, random_state=0).fit(XScale)
kmeans8.labels_
df['ClusterScr_km8'] = kmeans8.labels_

kmeans10 = KMeans(n_clusters=10, random_state=0).fit(XScale)
kmeans10.labels_
df['ClusterScr_km10'] = kmeans10.labels_


kmeans12 = KMeans(n_clusters=12, random_state=0).fit(XScale)
kmeans12.labels_
df['ClusterScr_km12'] = kmeans12.labels_

# 2.2 DBScan

# We try to visualize a good value of epsilon for dbscan parameterization 
# by computing the distance between nearest neighbour of the data
nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

distances =pd.DataFrame( {'distances': distances[:,1],
                              })
distances = distances.sort_values('distances',ascending=True)
distances = distances.reset_index(drop=True)

plt.plot(range(0, len(distances)),distances, marker = 'o')
plt.xlabel('Object')
plt.ylabel('k distance')
plt.show()
# The epsilon value around 20-30 should represent a distance between clusters

db = DBSCAN(eps=2, min_samples=10,metric='euclidean').fit(XScale)
df['Cluster_db'] = db.labels_
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
df.Cluster_db.unique()
# We get only 1 cluster from the dbscan method which may due to the fact that all the points are closely packed together.

# 2.3 Hierarchical agglomerative clustering with connectivity constraints

# we choose the ward linkage criteria for merging strategy (minimizing the sum of squared differences within all clusters)


connectivity = kneighbors_graph(XScale, n_neighbors=10, include_self=False)
ward = AgglomerativeClustering(n_clusters=7, connectivity=connectivity,
                               linkage='ward').fit(XScale)
df['ClusterScr_hclust7'] = ward.labels_

ward = AgglomerativeClustering(n_clusters=8, connectivity=connectivity,
                               linkage='ward').fit(XScale)
df['ClusterScr_hclust8'] = ward.labels_

ward = AgglomerativeClustering(n_clusters=10, connectivity=connectivity,
                               linkage='ward').fit(XScale)
df['ClusterScr_hclust10'] = ward.labels_


ward = AgglomerativeClustering(n_clusters=12, connectivity=connectivity,
                               linkage='ward').fit(XScale)
df['ClusterScr_hclust12'] = ward.labels_


# 2.3 2.4 Clustering result visualization

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.ClusterScr_km7)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()



threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.ClusterScr_km8)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.ClusterScr_km10)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.ClusterScr_km12)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.DIAGONAL_SCREEN_SIZE, c= df.ClusterScr_km12)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('SCREEN (inch)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.DIAGONAL_SCREEN_SIZE.min(),df.DIAGONAL_SCREEN_SIZE.max()])
threedee.view_init(15,15)
plt.show()

# kmean and hclust give similar clustering results

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.ClusterScr_hclust7)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()


threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.ClusterScr_hclust8)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.ClusterScr_hclust10)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df.PRICE_RELEASED, df.AGE, df.CAMERA_PIXELS, c= df.ClusterScr_hclust12)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
threedee.set_xlabel('PRICE (euros)')
threedee.set_ylabel('AGE (year)')
threedee.set_zlabel('CAMERA (pixels)')
threedee.set_xlim([df.PRICE_RELEASED.min(),df.PRICE_RELEASED.max()])
threedee.set_ylim([df.AGE.min(),df.AGE.max()])
threedee.set_zlim([df.CAMERA_PIXELS.min(),df.CAMERA_PIXELS.max()])
threedee.view_init(15,15)
plt.show()

df.head(2)

list(df.columns)

df.to_csv("DvcClusteringResults.csv", sep= ',', index=False)




