import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#import test and train file

train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# importing linear regressionfrom sklearn

from sklearn.linear_model import LinearRegression

# one model for linear regression, and another for polynomial
lreg = LinearRegression()
lregPoly = LinearRegression()


poly = PolynomialFeatures (degree=2)


# features for linear regression
X = train.loc[:,['Item_MRP']]

# features for poly regression
Xpoly=poly.fit_transform(X)



#splitting into training and cv for cross validation, for linear regression
x_train, x_cv, y_train, y_cv = train_test_split( X, train.Item_Outlet_Sales)

#splitting into training and cv for cross validation, for polynomial regression
x_trainP, x_cvP, y_trainP, y_cvP = train_test_split( Xpoly, train.Item_Outlet_Sales)


#training the linear model
lreg.fit (x_train, y_train)

#training the polynomial model
lregPoly.fit (x_trainP, y_trainP)

#predicting on cv:   polynomial
pred = lreg.predict (x_cv)

#predicting on cv:   linear
predP = lregPoly.predict (x_cvP)


#calculating mse
mse = np.mean((pred - y_cv)**2)
mseP = np.mean((predP - y_cvP)**2)

print ('\nlinear MSE=',round(mse,2))
print('linear R2=',r2_score (y_cv, pred))


print ('\nPoly MSE=',round(mseP,2))
print ('poly R2=',r2_score (y_cvP, predP))



# calculating coefficients
coeff = DataFrame(x_train.columns)

coeff['Coefficient Estimate'] = Series(lreg.coef_)


plt.plot(x_cv, y_cv,'.',color='blue')
plt.ylabel('MRP')

print ('\nLinear coefficients=',lreg.coef_)
print ('Linear intercept=',lreg.intercept_)

print ('\n---, a*x + b*x^2')
print ('Poly coefficients=',lregPoly.coef_)
print ('Poly intecept=',lregPoly.intercept_)


