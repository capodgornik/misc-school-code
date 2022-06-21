'''
April 2020
Math 496t
Explanation: CC† shows the direction of maximum variance in the data (matrix A) and the 2 component PCA is a rotation of those directions. 
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def randomCol(A, n):
    
    newA = np.zeros((2, n//2))
    index = np.random.randint(n, size = n//2)
    for i in range(n//2):
        newA[:,i] = A[:, index[i]]
    return newA

def reshapeMatrix(A1, A2, n):

    A = np.zeros((2, n*2))
    A[0][0:n] = A1[0]
    A[0][n:n*2] = A2[0]
    A[1][0:n] = A1[1]
    A[1][n:n*2] = A2[1]
    return A
    
def main():

    n = 100
    A1 = np.array(((np.random.normal(0, 1, n)),(np.random.normal(0, 0.1, n))))
    R = np.array(((np.cos(np.pi/2), -np.sin(np.pi/2)), (np.sin(np.pi/2), np.cos(np.pi/2))))
    A2 = R@A1
    A = np.array(((A1),(A2)))
    plt.scatter(A[0][0], A[0][1], label='Blob 1')
    plt.scatter(A[1][0], A[1][1], label ='Blob 2')
    pca = PCA(n_components = 2)
    A = reshapeMatrix(A1, A2, n)
    principalComponents = pca.fit_transform(np.transpose(A))
    x1 = np.array((0, pca.components_[0][0]))
    y1 = np.array((0, pca.components_[1][0]))
    x2 = np.array((0,pca.components_[0][1]))
    y2 = np.array((0, pca.components_[1][1]))
    plt.plot(x1,y1, color = 'green', label = 'PCA')
    plt.plot(x2,y2, color = 'green', label = 'PCA')
    C = randomCol(A, 2*n)
    cInv = np.linalg.pinv(C)
    projMatrix = np.dot(C, cInv)
    x3 = np.array((0, projMatrix[0][0]))
    y3 = np.array((0, projMatrix[1][0]))
    x4 = np.array((0, projMatrix[0][1]))
    y4 = np.array((0, projMatrix[1][1]))
    plt.plot(x3, y3, color = 'red', label = 'CC+')
    plt.plot(x4, y4, color = 'red', label = 'CC+')
    plt.legend()
    plt.show()
    
main()
