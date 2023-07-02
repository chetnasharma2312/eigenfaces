# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:54:23 2023

@author: Dell
"""

''' Find the mean of each row of Xsub and reshape as a 4096 × 1 vector '''
import numpy as np
import sklearn.datasets

faces = sklearn.datasets.fetch_olivetti_faces()
Xall = faces.data.T
print(Xall.shape)
print(Xall.min(), Xall.max())
Xsub = Xall[:,100:]

# Calculating the mean of each row of Xsub
mean_xsub = np.mean(Xsub, axis=1)
print('The mean of each row is: ',mean_xsub)

# Reshaping mean_xsub as a 4096 × 1 vector
xbar = mean_xsub.reshape(4096, 1)
print('The reshaped 4096 x 1 vector is: ',xbar)

''' Visualizing the “mean face” and comparing it to the image stored in the column of Xall'''

import matplotlib.pyplot as plt

# Plotting the mean face
plt.subplot(1,2,1)
plt.imshow(xbar.reshape(64,64), cmap=plt.cm.gray, vmin=0, vmax=1)
plt.title('Mean Face')

# Plotting the image corresponding to column 23 of Xall
plt.subplot(1,2,2)
plt.imshow(Xall[:, 23].reshape(64, 64), cmap=plt.cm.gray, vmin=0, vmax=1)
plt.title('Image from Xall Column 23')

plt.show()

''' Calculate the covariance matrix, find the eigenvalues (𝑉) and 
eigenvectors, confirm that P is an orthogonal matrix and that C = PDP^T '''
import numpy as np
import sklearn.datasets

faces = sklearn.datasets.fetch_olivetti_faces()
Xall = faces.data.T
Xsub = Xall[:,100:]

# Calculating the  4096 × 4096 covariance matrix of Xsub
cov_Xsub = np.cov(Xsub)
C = cov_Xsub.reshape(4096, 4096)
print('Covariance matrix of Xsub: ',C)

# Finding the eigenvalues (V) and eigenvectors (P) of C
V, P = np.linalg.eigh(C)
print('Eigenvalues(V) and Eigenvectors(P): ',V,P)

# Reversing the entries in V and the columns of P
V = V[::-1]
P = P[:, ::-1]

# Creating diagonal matrix D
D = np.diag(V)

# Confirming that P is an orthogonal matrix
print('P is an Orthogonal Matrix ?: ', np.allclose(np.dot(P.T, P), np.eye(P.shape[0])))

# Confirming that C = PDP^T
print('C = PDP^T ?: ',np.allclose(C, np.dot(np.dot(P, D), P.T)))

plt.figure()
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(P[:, i].reshape(64, 64), cmap=plt.cm.gray)
    plt.title(f"{i+1}")
    plt.axis('off')
plt.show()

Ysub = np.dot(P.T, Xsub-xbar [:, np.newaxis])

#Selecting column of Ysub
y = Ysub[:,0]

# Recreating corresponding column of Xsub
x_new = P @ y + xbar

# Plotting faces
plt.subplot(1, 2, 1)
plt.imshow(Xsub[:,0].reshape(64,64), cmap=plt.cm.gray)
plt.title('Actual Face')
plt.subplot(1, 2, 2)
plt.imshow(x_new.reshape(64,64), cmap=plt.cm.gray)
plt.title('New Face')
plt.show()

# Comparing
print(np.allclose(x, x_new))