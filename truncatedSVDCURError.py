'''
April 2020
Math 496t
Explanation: According to the output, the truncated SVD has a very large error, 
even though 98 percent of the spectral information of A is contained within i 
(in my case, i was 4) singular values. The error of A − CU†R and A − CC†AR†R were 
also fairly large, although the error of the latter was smaller. Letting C = A, 
the output showed that the larger the number of rows of A chosen for R, the lower 
the error. Furthermore, except for the last error, the errors for part F decreased 
approximately by ¼ each time, which was the inverse of the value of i. The error for 
A − CU†R and A − CC†AR†R was the same for each number of rows in part F. 
'''

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

def createMatrixFromData(data):
    
    newData = data.split('\n')
    n = len(newData)
    A = np.zeros((n, 14))
    for i in range(n):
        A[i,:] = newData[i].split(',')
    return(A)

def createSkeePlot(matrix):

    U,S,V = linalg.svd(matrix)
    x = np.zeros(len(S))
    y = np.zeros(len(S))
    for i in range(len(S)):
        y[i] = S[i]
        x[i] = i
    plt.plot(x,y)
    plt.xlabel('i')
    plt.ylabel('Singular Value')
    plt.title('Singular Values vs i')
    plt.show()
    return(U,S,V)
    
def findI(matrix, S):

    sum1 = 0
    for i in range(np.linalg.matrix_rank(matrix)):
        sum1 += S[i]
    sum2 = 0
    for j in range(len(S)):
        sum2 += S[j]
        if 100*(sum2/sum1) >= 98:
            break
    return(j)      
    
def findErrorTruncSVD(A, i):
    
    svd = TruncatedSVD(n_components = i)
    Ai = svd.fit_transform(A)
    Ai = np.pad(Ai, ((0,0), (0,10)), 'constant')
    return(np.linalg.norm(A-Ai), Ai)
    
def randomCol(A,n, transpose):
    if not transpose:
        newA = np.zeros((297, n))
    if transpose:
        newA = np.zeros((14, n))
    index = np.random.randint(n, size = n)
    for i in range(n):
        newA[:,i] = A[:, index[i]]
    return newA
  
def findU(A, C,R,k):

    U = (R@np.linalg.pinv(A)@C)
    return U
    
def findErrorPartE(A, i):
    
    error2 = 0
    error3 = 0
    for w in range(100):
        C = randomCol(A,i,transpose = False)
        R = randomCol(np.transpose(A),i, transpose = True)
        U = np.transpose(R)@np.linalg.pinv(A)@C
        error2 += linalg.norm(A - (C@linalg.pinv(U)@np.transpose(R)))
        error3 += linalg.norm(A - (C@linalg.pinv(C)@A@linalg.pinv(np.transpose(R))@np.transpose(R)))
    return error2/100, error3/100
        
def findErrorPartF(A, i):
    
    error4 = 0
    error5 = 0
    for z in range(100):
        C = A
        R = randomCol(np.transpose(A),i, transpose = True)
        U = np.transpose(R)@np.linalg.pinv(A)@C
        error4 += linalg.norm(A - (C@linalg.pinv(U)@np.transpose(R)))
        error5 += linalg.norm(A - (C@linalg.pinv(C)@A@linalg.pinv(np.transpose(R))@np.transpose(R)))
    return error4/100, error5/100
    
def partF(A, i):    

    y = 1
    for d in range(4):
        error4, error5 = findErrorPartF(A, y*i)
        print('Part F.1, number of rows = ' + str(y*i) + ':')
        print(error4)
        print('Part F.2, number of rows = ' + str(y*i) + ':')
        print(error5)
        if y == 1:
            y += 1
        elif y > 1:
            y += 2
            
def main():
    
    file = open("processed.cleveland.data")
    data = file.read()
    A = createMatrixFromData(data)
    U,S,V = createSkeePlot(A)
    i = findI(A, S)
    error, Ai = findErrorTruncSVD(A, i)
    print('Truncated SVD Error:')
    print(error)
    error2, error3 = findErrorPartE(A, i)
    print('Part E.1 Error:')
    print(error2)
    print('Part E.2 Error:')
    print(error3)
    partF(A, i)
    file.close()

main()	
