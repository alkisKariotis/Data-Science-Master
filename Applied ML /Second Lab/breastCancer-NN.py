#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 05:34:19 2018

@author: acggs

Test multiple classifiers on a dataset
"""
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn import datasets 
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import numpy as np


data=pd.read_csv('wpbc.data',header=None)

y=data[1]


# N, R are the two classes
y.replace('N',1,inplace=True)
y.replace('R',0,inplace=True)


x=data.iloc[:,2:]

# the following is a ?, i.e. a missing value
#x.iloc[196][34]
#replace it with a numpy nan
x=x.replace('?',np.NaN)

#x[34].loc[196]=x[34].median()
x[34].fillna(x[34].median(),inplace=True)


# make all columns with 0-mean, and 1-std
x = preprocessing.scale(x)


# split thedat sets into training and tests
x_train, x_test, y_train, y_test = train_test_split (x,y, test_size=0.3, random_state=1)

#Define Neural Network
#logistic', the logistic sigmoid function,
#          returns f(x) = 1 / (1 + exp(-x))
# 'relu', the rectified linear unit function,
#         returns f(x) = max(0, x)

#  solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
#
# batch_size: the size of the batch (number of data items used to compute error)
#
#early_stopping : bool, default=False
#    Whether to use early stopping to terminate training when validation
#    score is not improving. If set to true, it will automatically set
#    aside 10% of training data as validation and terminate training when
#    validation score is not improving by at least ``tol`` for
#    ``n_iter_no_change`` consecutive epochs.
#
clfANN = MLPClassifier(solver='adam', activation='relu',
                       batch_size=10,
                       tol=1e-7,
                       validation_fraction=0.2,
                       hidden_layer_sizes=(15,10,5,3), random_state=1, max_iter=50000, verbose=True)



#train the classifiers
clfANN.fit(x_train, y_train)                         


#get predictions on the train set
y_train_pred_ANN=clfANN.predict(x_train)


#test the trained model on the test set
y_test_pred_ANN=clfANN.predict(x_test)



confMatrixTestANN=confusion_matrix(y_test, y_test_pred_ANN, labels=None)
confMatrixTrainANN=confusion_matrix(y_train, y_train_pred_ANN, labels=None)



print ('\n Conf matrix, Train Set, Neural Net')
print (confMatrixTrainANN)
print ()



print ('Conf matrix, Test Set, Neural Net')
print (confMatrixTestANN)
print ()



# Measures of performance: Precision, Recall, F1


print ('Neural Net: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_ANN, average='macro'))
print ('Neural Net: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_ANN, average='micro'))
print ('\n')
