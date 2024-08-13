#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 13:15:42 2018

@author: acggs
"""
print(__doc__)

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import mixture


#commpute kmeans

def gaussianMixture (x):
    bicAll=[]
    aicAll=[]
    logLikelihood=[]
    clustersAll=[]
    maxClusters=32
    silhouettesAll=[]
    
    for n in range(2, maxClusters):
        gmm = mixture.GaussianMixture(n_components=n, covariance_type='full').fit(x)
        logLikelihood.append(gmm.score(x))
        clustersAll.append(n)
 
        labels=gmm.predict(x)
        silhouetteScore=metrics.silhouette_score(x, labels)
        silhouettesAll.append(silhouetteScore)
    
        
    return logLikelihood, clustersAll, silhouettesAll
        



def kmeansClustering (x):
    inertiasAll=[]
    silhouettesAll=[]
    clustersAll=[]
    maxClusters=32

    for n in range(2,maxClusters):
        #print 'Clustering for n=',n
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(x)
        y_kmeans = kmeans.predict(x)

        #get cluster centers
        kmeans.cluster_centers_

        #evalute
        #print 'inertia=',kmeans.inertia_
        
        silhouette_values = silhouette_samples(x, y_kmeans)
        #print 'silhouette=', np.mean(silhouette_values)
    
        inertiasAll.append(kmeans.inertia_)
        silhouettesAll.append(np.mean(silhouette_values))    
        clustersAll.append(n)
        


    return  clustersAll, silhouettesAll, inertiasAll



# Compute DBSCAN
#eps and min_samples are the two parameress
def dbscanClustering (x1):
    epsAll=[]
    minSamples=[]
    silhouette_values=[]
    clustersAll=[]
    
    for i in range(1,11):
        for j in range(1,11):
            db = DBSCAN(eps=i, min_samples=j).fit(x1)
        
  
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters_ > 1 and n_clusters_ < len(x1):
                #print("DBSCAn Silhouette Coefficient: %0.3f, Number of clusters" % metrics.silhouette_score(x1, labels),  n_clusters_)
                epsAll.append(i)
                minSamples.append(j)
                silhouette_values.append(metrics.silhouette_score(x1, labels))
                clustersAll.append(n_clusters_)
    return clustersAll, silhouette_values
            






# Read data file: Breast Cancer
data=pd.read_csv('wpbc.data',header=None)


y=data[1]
y.replace('N',0,inplace=True)
y.replace('R',1,inplace=True)


x=data.loc[:,2:]

# the following is a ?
#x[34].loc[196]

#replace it with a numpy nan
x=x.replace('?',np.NaN)

#x[34].loc[196]=x[34].median()
x[34].fillna(x[34].median(),inplace=True)
#x[33].fillna(x[33].median()[0])
x1=x.astype(float)
y1=y.astype(int)

# make all columns with 0-mean, and 1-std
x1 = preprocessing.scale(x1)


#uncomment the next three lines to read: Iris.data
data=pd.read_csv('iris2.data',header=None)
x1=data.loc[:,0:3]
x1 = preprocessing.scale(x1)

# do the clusterings
clustersAll1, silhouette_values1=dbscanClustering (x1)
clustersAll2, silhouette_values2, inertiasAll2 = kmeansClustering(x1)
loglikelihood3, clustersAll3, silhouette_values3= gaussianMixture(x1)

plt.figure(1)

#silhouette: bigger values are better
plt.title('DBScan:')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Values')
plt.plot(clustersAll1, silhouette_values1,'*')

plt.figure(2)
plt.title('kmeans:')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Values')
plt.plot(clustersAll2, silhouette_values2,'*-')

plt.figure(3)
plt.title('Gaussian Mixture:')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Values')
plt.plot(clustersAll3, silhouette_values3,'*-')



plt.figure(4)
plt.title('Gaussian Mixture:')
plt.xlabel('Number of clusters')
plt.ylabel('log-likelihood')
plt.plot(clustersAll3, loglikelihood3,'*-')
