# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 09:35:24 2019

@author: abhishekpandey
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir("C:\\Users\\abhishekpandey\\Desktop")
# Importing the dataset
dataset = pd.read_csv('character.csv')
X = dataset.iloc[:, 1:23].values
y = dataset.iloc[:, 24].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
a = np.isnan(X_train)
X_train[a] = 0
X_train_scaled = sc.fit_transform(X_train)

b = np.isnan(X_test)
X_test[b] = 0
X_test_scaled = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_scaled)
c = np.isnan(X)
X[c] = 0  
y_pred_prob = classifier.predict_proba(X)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)