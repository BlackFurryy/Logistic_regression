# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 06:41:33 2018

@author: vishawar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_excel("microchip_QA.xlsx")

dataset = dataset.sample(frac=1)

#from sklearn.utils import shuffle
#dataset = shuffle(dataset)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,2].values

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)


#feature scaling - no need

#fit the model
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression()

from sklearn.svm import SVC

classifier = SVC(kernel="rbf")
classifier.fit(X_train,y_train,)

y_predict = classifier.predict(X_test)

#calculate sucess percentage
result = np.mean(y_predict==y_test)*100

from sklearn.metrics import confusion_matrix
conf = np.float64(confusion_matrix(y_test,y_predict))

precision = conf[0,0]/(np.sum(conf[0,:]))
recall = conf[0,0]/(np.sum(conf[:,0]))

F_score = (2*precision*recall)/(precision+recall)


#from matplotlib.pyplot import ListedColormap
pos_idx = np.where(y==1)
neg_idx = np.where(y==0)

plt.scatter(X[pos_idx,0],X[pos_idx,1],marker ="+",color="blue")
plt.scatter(X[neg_idx,0],X[neg_idx,1],marker ="x",color="red")


h = .02  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

plt.title("SVM ")
plt.xlabel("Test1 Score")
plt.ylabel("Test2 Score")
plt.legend()
plt.show()