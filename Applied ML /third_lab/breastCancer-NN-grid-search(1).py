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
from sklearn.model_selection import GridSearchCV

data=pd.read_csv('wpbc.data',header=None)

y=data[1]


# N, R are the two classes
y.replace('N',0,inplace=True)
y.replace('R',1,inplace=True)


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



# The parameters and parameters values upon which to perform grid search
parameter_space = {
    'hidden_layer_sizes': [(5,), (10,), (10,5)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'tol': [1e-7],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}



mlp = MLPClassifier(max_iter=200)

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)



clf.fit(x_train, y_train)

print('Best parameters found:\n', clf.best_params_)


# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
#clf = GridSearchCV(svc, parameters)


#train the classifiers
clf.fit(x_train, y_train)                         


#test the trained model on the test set
y_test_pred_ANN=clf.predict(x_test)



confMatrixTestANN=confusion_matrix(y_test, y_test_pred_ANN, labels=None)

print ('\n\nConf matrix Neural Net')
print (confMatrixTestANN)
print ()



# Measures of performance: Precision, Recall, F1


print ('Neural Net: Macro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_ANN, average='macro'))
print ('Neural Net: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_ANN, average='micro'))
print ('\n')
