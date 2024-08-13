# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 21:45:08 2018

@author: Dimitrios
"""
#linear regression, two input features 
#we wil predict: Item_Outlet_Sales


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

print ('features=',train.columns)
print ('\n')

# importing linear regressionfrom sklearn

from sklearn.linear_model import LinearRegression

lreg = LinearRegression()

# Column valus imputation: fill in missing values with the mean 
train['Item_Weight'].fillna((train['Item_Weight'].mean()), inplace=True)



X = train.loc[:,['Item_MRP']]
#X = train.loc[:,['Outlet_Establishment_Year','Item_MRP','Item_Weight']]


#splitting into training and cv for cross validation
x_train, x_cv, y_train, y_cv = train_test_split(X,train.Item_Outlet_Sales,test_size=0.5)

#training the model
lreg.fit(x_train,y_train)

#predicting on cv
pred = lreg.predict(x_cv)

#calculating mse
mse = np.mean((pred - y_cv)**2)


r2_score(y_cv, pred)

print ('linear MSE=',round(mse,2))
print ('R2=',round(r2_score(y_cv, pred),2))



# calculating coefficients
coeff = DataFrame(x_train.columns)

coeff['Coefficient Estimate'] = Series(lreg.coef_)

print (X.columns)
print ('Coefficients=',np.round(lreg.coef_,2))
print ('Intercept=',round(lreg.intercept_,2))
