
# importing basic libraries

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

#import test and train file
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

## training the model




train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# importing linear regressionfrom sklearn

lreg = LinearRegression()
lregPoly = LinearRegression()

poly = PolynomialFeatures(degree=2)


train['Outlet_Establishment_Year'] = 2013 - train['Outlet_Establishment_Year']
train['Item_Weight'].fillna((train['Item_Weight'].mean()), inplace=True)

#X = train.loc[:,['Item_MRP']]
X = train.loc[:,['Outlet_Establishment_Year','Item_MRP','Item_Weight']]


Xpoly=poly.fit_transform(X)






#splitting into training and cv for cross validation

x_train, x_cv, y_train, y_cv = train_test_split(X,train.Item_Outlet_Sales,test_size=0.5, random_state=42)

x_trainP, x_cvP, y_trainP, y_cvP = train_test_split(Xpoly,train.Item_Outlet_Sales,test_size=0.5, random_state=42)



#training the model
lreg.fit(x_train,y_train)

ridgeReg = Ridge(alpha=0.1)
ridgeReg.fit(x_train,y_train)

lassoReg = Lasso(alpha=0.1)
lassoReg.fit(x_train,y_train)


lregPoly.fit(x_trainP,y_trainP)


#predicting on cv
pred = lreg.predict(x_cv)
predRidge = ridgeReg.predict(x_cv)
predLasso = lassoReg.predict(x_cv)

predP = lregPoly.predict(x_cvP)


#calculating mse
mse = np.mean((pred - y_cv)**2)
mseP = np.mean((predP - y_cvP)**2)
mseRidge = np.mean((predRidge - y_cv)**2)
mseLasso = np.mean((predLasso - y_cv)**2)


print ('\nMSE=',mse)
print ('MSE Ridge=',mseRidge)
print ('MSE Lasso=',mseLasso)
print ('MSE Poly=',mseP)


print('\nlinear R2=',r2_score (y_cv, pred))
print ('ridge R2=',r2_score (y_cv, predRidge))
print ('lasso R2=',r2_score (y_cv, predLasso))
print ('poly R2=',r2_score (y_cvP, predP))


print('\nlinear coeff= ',lreg.coef_)
print('ridge coeff=',ridgeReg.coef_)
print('lasso coeff=',lassoReg.coef_)
print('Poly coeff= ' ,lregPoly.coef_)




