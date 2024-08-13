# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 21:45:08 2018

@author: Dimitrios
"""
#linear regression, two input features 
# importing basic libraries

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns



#import test and train file


def multicollinearity_assumption(residuals, features):
    """
    Multicollinearity: Assumes that predictors are not correlated with each other. If there is
                       correlation among the predictors, then either remove prepdictors with high
                       Variance Inflation Factor (VIF) values or perform dimensionality reduction
                           
                       This assumption being violated causes issues with interpretability of the 
                       coefficients and the standard errors of the coefficients.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    print('----------------------------------------------------------------')
    print('\n Assumption : Little to no multicollinearity among predictors')
        
    # Plotting the heatmap
    plt.figure(figsize = (10,8))
    sns.heatmap(pd.DataFrame(features).corr(), annot=True)
    plt.title('Correlation of Variables')
    plt.show()
        
    print('Variance Inflation Factors (VIF)')
    print('> 10: An indication that multicollinearity may be present')
    print('> 100: Certain multicollinearity among the variables')
    print('-------------------------------------')
       
    # Gathering the VIF for each variable
    VIF = [variance_inflation_factor(features, i) for i in range(features.shape[1])]
    for idx, vif in enumerate(VIF):
        print('idx, vif', idx, vif)
        
    # Gathering and printing total cases of possible or definite multicollinearity
    possible_multicollinearity = sum([1 for vif in VIF if vif > 10])
    definite_multicollinearity = sum([1 for vif in VIF if vif > 100])
    print()
    print('{0} cases of possible multicollinearity'.format(possible_multicollinearity))
    print('{0} cases of definite multicollinearity'.format(definite_multicollinearity))
    print()

    if definite_multicollinearity == 0:
        if possible_multicollinearity == 0:
            print('Assumption satisfied')
        else:
            print('Assumption possibly satisfied')
            print()
            print('Coefficient interpretability may be problematic')
            print('Consider removing variables with a high Variance Inflation Factor (VIF)')

    else:
        print('Assumption not satisfied')
        print()
        print('Coefficient interpretability will be problematic')
        print('Consider removing variables with a high Variance Inflation Factor (VIF)')





def normal_errors_assumption(residuals,  p_value_thresh=0.05):
    """
    Normality: Assumes that the error terms are normally distributed. If they are not,
    nonlinear transformations of variables may solve this.
               
    This assumption being violated primarily causes issues with the confidence intervals
    """
    from statsmodels.stats.diagnostic import normal_ad
    print('----------------------------------------------------------------')

    print('Assumption: The error terms are normally distributed', '\n')
    
    
    print('Using the Anderson-Darling test for normal distribution')

    # Performing the test on the residuals
     # Returns
    #-------
    #ad2 : float
    #    Anderson Darling test statistic.
    #pval : float
    #    The pvalue for hypothesis that the data comes from a normal
    #    distribution with unknown mean and variance.
        
    test, p_value = normal_ad(residuals)
    print('p-value from the test - below 0.05 generally means non-normal:', p_value)
    
    # Reporting the normality of the residuals
    if p_value < p_value_thresh:
        print('Residuals are not normally distributed')
    else:
        print('Residuals are normally distributed')
    
    # Plotting the residuals distribution
    plt.subplots(figsize=(12, 6))
    plt.title('Distribution of Residuals')
    #sns.histplot(residuals)
    sns.histplot(residuals, kde=True)


    plt.show()
    
    print()
    if p_value > p_value_thresh:
        print('Assumption satisfied')
    else:
        print('Assumption not satisfied')
        print()
        print('Confidence intervals will likely be affected')
        print('Try performing nonlinear transformations on variables')


def homoscedasticity_assumption(residuals):
    """
    Homoscedasticity: Assumes that the errors exhibit constant variance
    """
    print('\n----------------------------------------------------')
    print('Assumption: Homoscedasticity of Error Terms', '\n')
    
    print('Residuals should have relative constant variance')
    print('View the plot')    
  
    # Plotting the residuals
    plt.subplots(figsize=(12, 6))
    ax = plt.subplot(111)  # To remove spines
    plt.scatter(x=np.arange(0,len(residuals)), y=residuals, alpha=0.5)
    plt.plot(np.repeat(0, len(residuals)), color='darkorange', linestyle='--')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine
    plt.title('Residuals')
    plt.show()  




train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# importing linear regressionfrom sklearn

from sklearn.linear_model import LinearRegression

lreg = LinearRegression()

#splitting into training and cv for cross validation
train['Item_Weight'].fillna((train['Item_Weight'].mean()), inplace=True)

X = train.loc[:,['Outlet_Establishment_Year','Item_MRP','Item_Weight']]
# = train.loc[:,['Item_MRP']]

x_train, x_cv, y_train, y_cv = train_test_split(X,train.Item_Outlet_Sales,test_size=0.5)

#training the model

lreg.fit(x_train,y_train)

#predicting on cv

pred = lreg.predict(x_cv)

#calculating mse

mse = np.mean((pred - y_cv)**2)


r2_score(y_cv, pred)

print ('MSE=',mse)
print ('R2=',r2_score(y_cv, pred))



# calculating coefficients
coeff = DataFrame(x_train.columns)

coeff['Coefficient Estimate'] = Series(lreg.coef_)


print ('Coefficients=',lreg.coef_)
print ('Intercept=',lreg.intercept_)


residuals = (y_cv - pred)

#Test the assumption, about the normal distribution of errors
normal_errors_assumption(residuals)

#Test the assumption of homescedacity
homoscedasticity_assumption(residuals)

#test the assumption of multicollinearity
multicollinearity_assumption(residuals, np.array(X))

