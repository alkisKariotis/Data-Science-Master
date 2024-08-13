#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 11:00:06 2018

@author: acggs
"""

# Decision Trees as  classifiers for supervised machine learning


from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve
from sklearn import datasets 
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import random
from sklearn.model_selection import train_test_split, cross_val_score


from sklearn.feature_selection import RFE


#Read data from a file: The file is in CSV format
#class labels should be: 1,2,3
data=pd.read_csv('iris2.data',header=None)

#Define a tree classifier, with the appropriate parameters. 
# you can define no-parameters and it will take the default values
# 
#No train  takes place here
clf = tree.DecisionTreeClassifier(criterion="gini", random_state=0)

#data have 4 attributes, (first 4 columns)
X=data.iloc[:,0:5]

#data labels
Y=data.iloc[:,4]

print ('Short data report:')
print('#classes=',len(Y.unique()))
a=Y.groupby(Y).count()
print('Class-1 #number of samples=',a[1])
print('Class-2 #number of samples=',a[2])    
print('Class-3 #number of samples=',a[3])


#convert pandas data to numpy array
X=np.array(X)

#print dimensions of the data
nRows, nCols = X.shape
print ('#Data samples=',nRows)
print ('#Data attributes=',nCols-1)   
print('------------------------------------------')
print()



# Percentage of data to use in training: 50%
# The rest will be used for testing



####  Data set separated intro training and testing ####
# Training set to build the classifier will be 50% of the data
# number of data attribute=4 (indexed from 0 to 3)
 #class label is the last column (indexed by 4)


train_size=0.5  # 50 per cent of the data
print("Percentage used in training:", train_size)

X_train, X_test, Y_train, Y_test = train_test_split (X[:,0:4], X[:,4], train_size=0.50, random_state=0)



###  Training #####
#Actual training of the classifier
#training data followed by labels
clf = clf.fit(X_train, Y_train)

###  Tesing #####
#test the trained model on the training set
Y_train_pred=clf.predict(X_train)

#test the trained model on the test set
Y_test_pred=clf.predict(X_test)

#produce confusin matrix on the training data
confMatrixTrain=confusion_matrix(Y_train, Y_train_pred, labels=None)

#produce confusin matrix on the TESTING data
confMatrixTest=confusion_matrix(Y_test, Y_test_pred, labels=None)



### Evaluation ###
# Evaluation on accuracy: training, testing set
print ('\tClassifier Evaluation')
print ('Accuracy Train=', accuracy_score(Y_train, Y_train_pred, normalize=True))
print ('Accuracy Test=', accuracy_score(Y_test, Y_test_pred, normalize=True))
print()

# Evaluation on Confusion matrix: training, testing set
print ('Confusion matrix train')
print (confMatrixTrain,'\n')
print ('Confusion matrix test')
print (confMatrixTest)
print ()

# Measures of performance: Precision, Recall, F1
# Macro: 
print('Macro-train f1=',f1_score(Y_train, Y_train_pred, average='macro')) 
print('Macro-test f1=',f1_score(Y_test, Y_test_pred, average='macro')) 
print()
# Micro: 
print('Micro-train f1=',f1_score(Y_train, Y_train_pred, average='micro')) 
print('Micro-test f1=',f1_score(Y_test, Y_test_pred, average='micro')) 
print()
print('train f1 per class=',f1_score(Y_train, Y_train_pred, average=None))
print('test_f1 per class =',f1_score(Y_test, Y_test_pred, average=None)) 
print()
#
print ('Macro: train-Precision-Recall-FScore-Support',precision_recall_fscore_support(Y_train, Y_train_pred, average='macro'))
print ('Macro: test-Precision-Recall-FScore-Support',precision_recall_fscore_support(Y_test, Y_test_pred, average='macro'))
print ('\n')
print ('\n')


#the Decision tree is stored in the current directory
# to see the actual tree paste to .dot file to
# http://www.webgraphviz.com/  
tree.export_graphviz(clf, out_file='iris-tree.dot',class_names=['setosa','versicolor','virginica'])
#
print('Tree rules=\n',tree.export_text(clf,feature_names=['sepal_length','sepal_width','petal_length','petal_width']))
# Some statistics on the Tree:
print('Max Depth=',clf.get_depth())
print('Number of Leaves=',clf.get_n_leaves())
