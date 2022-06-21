'''
April 2020
Math 496t
Explanation: I used a 2,500 by 2,500 matrix. The output is in units of seconds. 
Parts A and B showed that computing the truncated SVD takes less time than computing 
the full rank SVD. In particular, with a rank 1/3 that of the rank of A the time was 
2/3 shorter. Parts C and D showed that computing CU†R takes longer than computing CC†AR†R. 
It seems that computing U takes a lot more time than computing C and R and their inverses. 
Parts E and F showed that taking the partial QR decomposition takes less time than computing 
the full QR decomposition. In particular, it takes about 15 times as long for the full QR 
decomposition. 
'''

import numpy as np
import arrow
from sklearn.decomposition import TruncatedSVD
import random
from scipy import linalg

def partA(A):

    startTime = arrow.now()
    U, S, V = np.linalg.svd(A)
    endTime = arrow.now()
    return(endTime-startTime)

def partB(A, rank):

    startTime = arrow.now()
    svd = TruncatedSVD(n_components = rank)
    svd.fit(A)
    endTime = arrow.now()
    return(endTime-startTime)

def randomizeMatrix(A, size):
    
    indices = np.zeros(size)
    for i in range(size):
        rowIndices[i] = np.random.randint(size)  
    C = np.zeros((size,size))
    R = np.zeros((size,size))
    for i in range(size):
        index = int(indices[i])
        C[:, i] = A[0:833, indexC]
        R[i, :] = A[indexR, 0:833]
    return(C, R)
  
def selectCol(A, size):
        
    indices = np.zeros(size)
    for i in range(size):
        indices[i] = np.random.randint(size)  
    C = np.zeros((size,size))
    for i in range(size):
        index = int(indices[i])
        C[:, i] = A[0:size, index]
    return(C)
  
def partC(A, size):
    
    startTime = arrow.now()
    C = selectCol(A, size)
    R = selectCol(np.transpose(A), size)
    U = np.transpose(R)@linalg.pinv(A[:833,:833])@C
    C@linalg.pinv(U)@np.transpose(R)
    endTime = arrow.now()
    return(endTime-startTime)
    
def partD(A, size)    :

    startTime = arrow.now()
    C = selectCol(A, size)
    R = selectCol(np.transpose(A), size)
    C@linalg.pinv(C)@A[:833,:833]@linalg.pinv(R)@R
    endTime = arrow.now()
    return(endTime-startTime)
    
def partE(A):

    startTime = arrow.now()
    Q, R = np.linalg.qr(A)
    endTime = arrow.now()
    return(endTime-startTime)
    
def partF(A):
    
    startTime = arrow.now()
    Q, R = np.linalg.qr(A[:833,:833])
    endTime = arrow.now()
    return(endTime-startTime)
    
def main():
    
    A = np.random.rand(2500,2500)
    print('Part A:')
    print(partA(A))
    print('Part B:')
    print(partB(A, 833))
    print('Part C:')
    print(partC(A, 833))
    print('Part D:')
    print(partD(A, 833))
    print('Part E:')
    print(partE(A))
    print('Part F:')
    print(partF(A))
    
main()
