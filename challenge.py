# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:33:58 2020

@author: João Gabriel Andrade
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Read csv files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# List all test insc and if they were present on the second day
answer = test[['NU_INSCRICAO', 'TP_PRESENCA_LC']]

# Drop NU_INCRICAO
inscTrain = train['NU_INSCRICAO']
train.drop('NU_INSCRICAO', axis=1, inplace=True)
inscTest = test['NU_INSCRICAO']
test.drop('NU_INSCRICAO', axis=1, inplace=True)

# Drop all columns that is not present in test df except the target
train.drop(train.columns.difference(test.columns.tolist() + ['NU_NOTA_MT']), axis=1, inplace=True)

# Drop irrelevant columns
irrelevantColumns = ['CO_PROVA_CH', 'CO_PROVA_CN', 'CO_PROVA_LC', 'CO_PROVA_MT', 'SG_UF_RESIDENCIA']
train.drop(irrelevantColumns, axis=1, inplace=True)
test.drop(irrelevantColumns, axis=1, inplace=True)

# Convert string columns to int
trainColumns = train.select_dtypes(['object']).columns
testColumns = test.select_dtypes(['object']).columns
trainDummy = pd.get_dummies(trainColumns)
testDummy = pd.get_dummies(testColumns)
train.drop(trainColumns, axis = 1, inplace=True)
test.drop(testColumns, axis = 1, inplace=True)
train = pd.concat([train, trainDummy], axis = 1)
test = pd.concat([test, testDummy], axis = 1)

# Remove the target
target = train['NU_NOTA_MT']
train.drop('NU_NOTA_MT', axis=1, inplace=True)

# Convert 'NaN' to 0
train = train.fillna(0)
target = target.fillna(0)
test = test.fillna(0)

# Generate Polynomial features 
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(train)
test = poly.fit_transform(test)

# Split df(70% train and 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_poly, target, random_state = 0)

#Fit the model
linreg = LinearRegression().fit(X_train, y_train)

print('(poly deg 2) linear model intercept (b): {:.3f}'
    .format(linreg.intercept_))
print('(poly deg 2) R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('(poly deg 2) R-squared score (test): {:.3f}\n'
    .format(linreg.score(X_test, y_test)))

# Predict test df
predict = linreg.predict(test)
answer['NU_NOTA_MT'] = predict

#Grades has to be at least 309.7(min grade in math - enem 2016) and max of 991.5(max grade in math - enem 2016)
minGrade = 309.7
answer.loc[answer['NU_NOTA_MT'] < minGrade, 'NU_NOTA_MT'] = minGrade
maxGrade = 991.5
answer.loc[answer['NU_NOTA_MT'] > maxGrade, 'NU_NOTA_MT'] = maxGrade

#0 to people that were not present
answer.loc[answer['TP_PRESENCA_LC'] != 1, 'NU_NOTA_MT'] = 0

#Convert to csv file
answer.drop('TP_PRESENCA_LC', axis=1, inplace=True)
answer.to_csv('answer.csv', sep=',', index=False)


