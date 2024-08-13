#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 19:41:05 2018

@author: acggs
"""
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans


#2021, from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_blobs


#generate some random data
#change the: number of samples: 'n_smaples'
# number of 'centers', 'cluster_std' is the standard deviation
#to see the difference
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

#plot them
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], s=50)



#set a different number of clusters
# and see how they affect the 'inertia' and the 'silhouette'
kmeans = KMeans(n_clusters=4)


# perform clustering
kmeans.fit(X)

# find the clusters the points X belong to
y_kmeans = kmeans.predict(X)


#get cluster centers
kmeans.cluster_centers_

#evalute clustering
#inertia: lower values are better
#Silhouette: higher values are better
print ('inertia=',kmeans.inertia_)
print ('inertia=',round(kmeans.inertia_,3))
silhouette_values = silhouette_samples(X, y_kmeans)
print ('silhouette=', np.mean(silhouette_values))
print ('silhouette=', round(np.mean(silhouette_values),3))



# plot sample colored according to the cluster they belong to
plt.figure(2)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=400, alpha=0.2);
