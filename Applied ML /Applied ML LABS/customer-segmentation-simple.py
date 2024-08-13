#source: https://towardsdatascience.com/clustering-algorithms-for-customer-segmentation-af637c6830ac
#date: 1st March 2020

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt#Plot styling
import seaborn as sns; sns.set()  # for plot styling
from sklearn.cluster import KMeans


# size of figure in inches
plt.rcParams['figure.figsize'] = (16, 9)

#read the dataset
dataset=pd.read_csv('CLV.csv')

#top 5 columns
dataset.head()

# of rows#descriptive statistics of the dataset
len(dataset) 
dataset.describe().transpose()

plt.figure(1)
#Visualizing the data - displot
plot_income = sns.distplot(dataset["INCOME"])
plot_spend = sns.distplot(dataset["SPEND"])
plt.xlabel('Income / spend')



plt.figure(2)
X=np.array(dataset)
#Using the elbow method to find the optimum number of clusters
wcss = []
for i in range(1,11):
    km=KMeans(n_clusters=i, random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()


##Fitting kmeans to the dataset with k=4
km4=KMeans(n_clusters=4, random_state=0)
y_means = km4.fit_predict(X)#Visualizing the clusters for k=4

# s=50, its the size of the dots
plt.scatter(X[y_means==0,0],X[y_means==0,1],s=50, c='purple',label='Cluster1')
plt.scatter(X[y_means==1,0],X[y_means==1,1],s=50, c='blue',label='Cluster2')
plt.scatter(X[y_means==2,0],X[y_means==2,1],s=50, c='green',label='Cluster3')
plt.scatter(X[y_means==3,0],X[y_means==3,1],s=50, c='cyan',label='Cluster4')
plt.scatter(km4.cluster_centers_[:,0], km4.cluster_centers_[:,1],s=200,marker='s', c='red', alpha=0.7, label='Centroids')
plt.title('Customer segments')
plt.xlabel('Annual income of customer')
plt.ylabel('Annual spend from customer on site')
plt.legend()
plt.show()

##Fitting kmeans to the dataset - k=6
km4=KMeans(n_clusters=6,random_state=0)
y_means = km4.fit_predict(X)#Visualizing the clusters
plt.scatter(X[y_means==0,0],X[y_means==0,1],s=50, c='purple',label='Cluster1')
plt.scatter(X[y_means==1,0],X[y_means==1,1],s=50, c='blue',label='Cluster2')
plt.scatter(X[y_means==2,0],X[y_means==2,1],s=50, c='green',label='Cluster3')
plt.scatter(X[y_means==3,0],X[y_means==3,1],s=50, c='cyan',label='Cluster4')
plt.scatter(X[y_means==4,0],X[y_means==4,1],s=50, c='magenta',label='Cluster5')
plt.scatter(X[y_means==5,0],X[y_means==5,1],s=50, c='orange',label='Cluster6')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],s=200,marker='s', c='red', alpha=0.7, label='Centroids')
plt.title('Customer segments')
plt.xlabel('Annual income of customer')
plt.ylabel('Annual spend from customer on site')
plt.legend()
plt.show()