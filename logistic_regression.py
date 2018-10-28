# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 11:33:11 2018

@author: vishawar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load data set

dataset = pd.read_excel("university_exam.xlsx")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,2].values

#separate train & test data

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)


#feature scaling 

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)

#fit a logistic regression model over dataset

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

#from sklearn.svm import SVC
#classifier = SVC()

classifier.fit(X_train,y_train)

#predict
y_predict = classifier.predict(X_test)
result_percentage = np.mean(y_predict==y_test)*100

#confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test,y_predict)

#from matplotlib.pyplot import ListedColormap
#pos_idx = np.where(y==1)
#neg_idx = np.where(y==0)

#plt.scatter(X[pos_idx,0],X[pos_idx,1],marker ="+",color="blue")
#plt.scatter(X[neg_idx,0],X[neg_idx,1],marker ="x",color="red")

from matplotlib.colors import ListedColormap
h = .02  # step size in the mesh
# create a mesh to plot in
X_set=X_train
y_set = y_train
x_min, x_max = X_set[:, 0].min() - 1, X_set[:, 0].max() + 1
y_min, y_max = X_set[:, 1].min() - 1, X_set[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

plt.contourf(xx,yy,classifier.predict(np.array([xx.ravel(),yy.ravel()]).T).reshape(xx.shape),alpha=0.75,cmap=ListedColormap(("red","green")))
plt.xlim(xx.min(),xx.max())
plt.ylim(xx.min(),xx.max())


for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j,1],c= ListedColormap(("Green","red"))(i),label =j)
    
    
plt.legend()