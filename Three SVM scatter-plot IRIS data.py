# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 06:14:10 2022

@author: Aparajita
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import datasets

irisset = datasets.load_iris()
X = irisset.data[:100,:2]
z = irisset.target[:100]

X1= irisset.data[50:150,:2]
z1 = irisset.target[50:150]

X2=np.concatenate((irisset.data[:50,:2], irisset.data[100:150,:2]))
z2=np.concatenate((irisset.target[:50], irisset.target[100:150]))


X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.3)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, z1, test_size=0.3)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, z2, test_size=0.3)

svc_model= SVC()
svc_model.fit(X_train, y_train)

#svc_model1= SVC()
svc_model.fit(X1_train, y1_train)

#svc_model2= SVC()
svc_model.fit(X2_train, y2_train)

clf = SVC(kernel='linear',C=1.0)
clf.fit(X,z)
w=clf.coef_[0]

plt.figure(figsize=(10, 4))
#plt.pcolormesh(X, z, shading='auto')
xpoints = np.linspace(4.5,6.5)
ypoints = -w[0] / w[1] * xpoints -clf.intercept_[0]/ w[1]
plt.plot(xpoints, ypoints, 'g-')
plt.scatter(X[:, 0], X[:, 1], c = z, marker='o', cmap='Greens')


clf=SVC(kernel='linear',C=1.0)
clf.fit(X1,z1)
w1=clf.coef_[0]
x1points = np.linspace(5,7)
y1points = -w1[0] / w1[1] * x1points -clf.intercept_[0]/ w1[1]
plt.plot(x1points, y1points, 'r-')
plt.scatter(X1[:, 0], X1[:, 1], c = z1, alpha=.5, cmap='BuGn_r')


clf=SVC(kernel='linear',C=1.0)
clf.fit(X2,z2)
w2=clf.coef_[0]
x2points = np.linspace(4.5,6.5)
y2points = -w2[0] / w2[1] * x2points -clf.intercept_[0]/ w2[1]
plt.plot(x2points, y2points, 'b-')
plt.scatter(X2[:, 0], X2[:, 1], c = z2, alpha= .5, cmap= 'pink')


plt.suptitle('SVM IRIS Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()