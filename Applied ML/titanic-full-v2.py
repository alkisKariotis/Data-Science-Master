# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 22:12:06 2019

@author: dimitrv
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report



def handle_categorical_na(df):
    ## Imputing the null/na/nan values in 'Age' attribute with its mean value 
    df.Age.fillna(value=df.Age.mean(),inplace=True)
    
    
    ## replacing the null/na/nan values in 'Embarked' attribute with 'X'
    df.Embarked.fillna(value='X',inplace=True)
    return df




### READ data ###



#    survival - Survival (0 = No; 1 = Yes)
#    class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
#    name - Name
#    sex - Sex
#    age - Age
#    sibsp - Number of Siblings/Spouses Aboard
#    parch - Number of Parents/Children Aboard
#    ticket - Ticket Number
#    fare - Passenger Fare
#    cabin - Cabin
#    embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
#    boat - Lifeboat (if survived)
#    body - Body number (if did not survive and body was recovered)


trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

trainData.head()
trainData.describe()



## PREPROCESSING ##

# for each column report the percentage of data that are missing
pd.DataFrame({'percent_missing': trainData.isnull().sum() * 100 / len(trainData)})


# Names of the features extacted from the data
selFeatures = list(trainData.columns.values)

# Removing the target variable from the column values
# This is the feature we are trying to predict
targetCol = 'Survived'
selFeatures.remove(targetCol)

# Removing features with unique values
for i in selFeatures:
    if trainData.shape[0] == len(pd.Series(trainData[i]).unique()) :
        selFeatures.remove(i)
        
# Removing features with high percentage of missing values
selFeatures.remove('Cabin')
        





### VISUALIZATION ###

#plot the passangers that survived per sex
sex = pd.crosstab([trainData.Survived], trainData.Sex)
sex.plot.bar()

# Also removing cabin and ticket features for the initial run.
selFeatures.remove('Ticket')
        
print("Target Class: '"+ targetCol + "'")
print('Features to be used in prediction: ')
print(selFeatures)






### TRAINING ###

seed = 7
np.random.seed(seed)

#percentage for testing
perc=0.3
X_train, X_test, Y_train, Y_test = train_test_split(trainData[selFeatures], trainData.Survived, test_size=perc)

X_train = handle_categorical_na(X_train)
X_test = handle_categorical_na(X_test)


## using One Hot Encoding for handling categorical data
X_train = pd.get_dummies(X_train,columns=['Embarked','Sex'],prefix=['Embarked','Sex'])
X_test = pd.get_dummies(X_test,columns=['Embarked','Sex'],prefix=['Embarked','Sex'])

featureNames=list(X_train.columns)


## find the common columns in the test and train data sets
common_col = [x for x in X_test.columns if x in X_train.columns]
X_test = X_test[common_col]

missing_col = [x for x in X_train.columns if x not in X_test.columns]
## Inserting missing columns in test data
for val in missing_col:
    X_test.insert(X_test.shape[1], val, pd.Series(np.zeros(X_test.shape[0])))
    
    
# Covert data to numpy arrays
X_train=np.array(X_train)
Y_train=np.array(Y_train)
X_test=np.array(X_test)
Y_test=np.array(Y_test)


#Training the classifier
#Define decision Tree

clfDT =  tree.DecisionTreeClassifier()
#clfDT =  tree.DecisionTreeClassifier( max_depth=None, min_samples_leaf=20)
clfDT.fit(X_train, Y_train)



#test the trained model on the training set
Y_train_pred_DT=clfDT.predict(X_train)
#test the trained model on the test set
Y_test_pred_DT=clfDT.predict(X_test)







### EVALUATE the classifer  ###

confMatrixTrainDT=confusion_matrix(Y_train, Y_train_pred_DT, labels=None)
confMatrixTestDT=confusion_matrix(Y_test, Y_test_pred_DT, labels=None)


print ('train: Conf matrix Decision Tree')
print (confMatrixTrainDT)
print ()

print ('test: Conf matrix Decision Tree')
print (confMatrixTestDT)
print ()


pr_y_test_pred_DT=clfDT.predict_proba(X_test)

#ROC curve for the class encoded by 1, survived
fprDT, tprDT, thresholdsDT = roc_curve(Y_test, pr_y_test_pred_DT[:,1])


print(classification_report(Y_test, Y_test_pred_DT))



#line width
lw=2
plt.figure(10)
plt.plot(fprDT,tprDT,color='blue')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()



dot_data = tree.export_graphviz(clfDT, out_file=None) 
tree.export_graphviz(clfDT, out_file='tree.dot')



print('Tree rules=\n',tree.export_text(clfDT,feature_names=featureNames))
# Some statistics on the Tree:
print('Max Depth=',clfDT.get_depth())
print('Number of Leaves=',clfDT.get_n_leaves())


