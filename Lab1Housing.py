#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 13:23:49 2017
Machine Learning Lab 1

@author: Tommy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
names =[
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
    'AGE',  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE'
]

# TODO:  Complete the code

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',header='infer',delim_whitespace=True,names=names,na_values='?')
print(df.head(6))


print("Shape: " , df.shape)
print("Number of samples: " + str(df.shape[0]) +  ". Number of attributes: " + str(df.shape[1]) + '.')

y  = np.array(df["PRICE"])

print("\nThe mean house price is " + str(np.mean(y)) + " thousand dollars\n")
above_40 = (y > 40)

print("Only " + str((len(y[above_40]) / len(y)) * 100) + " percent are above $40k.\n")


x = np.array(df["RM"])

plt.plot(x,y,'o')
plt.grid()

def fit_linear(x,y):
    """
    Given vectors of data points (x,y), performs a fit for the linear model:
       yhat = beta0 + beta1*x, 
    The function returns beta0, beta1 and rsq, where rsq is the coefficient of determination.
        """    
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    rsq = r_value ** 2 
    return intercept, slope, rsq
beta0 , beta1, rsq = fit_linear(x,y)

xplt = np.array([4,9])          
yplt = beta1*xplt + beta0
plt.plot(xplt,yplt,'-',linewidth=3)  # Plot the regression line
plt.xlabel("Rooms")
plt.ylabel("Price")

print('{:10}'.format("Attribute") + '{:10}'.format("R^2"))
for name in names:
    if name == "PRICE":
        break
    x = df[name]
    beta0,beta1,rs1 = fit_linear(x,y)
    #These are the coefficent of determinations
    print('{:10}'.format(name) + '{:10}'.format(rs1))
 


