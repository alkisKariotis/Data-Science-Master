import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

#diabetes.feature_names
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
#diabes.DESCR
#bp= blood presure, and 6 blood serum movements
#predict disease progressing one uear after base line



diabetes_X = diabetes.data

# Split the data into training/testing sets
nTest=100
diabetes_X_train = diabetes_X[:-nTest]
diabetes_X_test = diabetes_X[-nTest:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-nTest]
diabetes_y_test = diabetes.target[-nTest:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Create a polynomial-regression object
# Observe that they are identical
regr2 = linear_model.LinearRegression()


#regr2 = Ridge(alpha=2)
#regr2=Lasso(alpha=0.01)


#This creates the features for the polynomail
polyDegree=2
poly = PolynomialFeatures(degree=polyDegree)


#transform the data to make the suitable for polynomial regression
a=poly.fit_transform(diabetes_X_train)
a2=poly.fit_transform(diabetes_X_test)
#[1, a, b, a^2, ab, b^2].

regr.fit(diabetes_X_train, diabetes_y_train)
# Create a polynomial-regression object
regr2.fit(a, diabetes_y_train)


# Make predictions using the testing set, linear
diabetes_y_pred = regr.predict(diabetes_X_test)
# Make predictions using the testing set, polynomial
diabetes_y_pred_poly = regr2.predict(a2)


# The coefficients
print ('Coefficients Linear Regression: \n', np.round(regr.coef_,2))
print ('Intercept Linear Regression: ', np.round(regr.intercept_,2))

# The mean squared error
"Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred)
# Explained variance score: 1 is perfect prediction
print ('R2 score Linear: %.2f'  % r2_score(diabetes_y_test, diabetes_y_pred))
print ("Mean absolute error Linear: %.2f" % mean_absolute_error(diabetes_y_test, diabetes_y_pred))
print ("Mean squared error Linear: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))


regr.intercept_
regr.coef_






# Display The coefficients
print ('\n')
print ('Coefficients regr2 Poly Degree:',str(polyDegree) ,'\n', (regr2.coef_))
print ('Intercept regr2 Poly Degree:',(round(regr2.intercept_,2)))

print ('R2 score Poly: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred_poly))
# The mean squared error
print ("Mean squared error Poly: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred_poly))
# Explained variance score: 1 is perfect prediction
print ("Mean absolute error Poly: %.2f" % mean_absolute_error(diabetes_y_test, diabetes_y_pred_poly))

