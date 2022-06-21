'''
April 2020
Math 496t
Explanation: To find each accuracy, I ran the k-Nearest Neighbor algorithm 
100 times and took the average. The accuracies for the 1-Nearest Neighbor and 
2-Nearest Neighbor algorithms ended up being very similar. The 2-Nearest 
Neighbor algorithm was slightly less accurate than the 1-Nearest Neighbor 
algorithm, however.  
'''

import math 
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import random

def findKNeighbors(data, k, x):

    neighbors = data
   
    for j in range(len(data)):
        distance = 0
        for i in range(len(data[j])-1):
            distance += (data[j][i] - x[i])**2
        neighbors[j].append(math.sqrt(distance))
    neighbors.sort(key = getLast)
    [k.pop(-1) for k in neighbors]
    
    return neighbors[:][:k]
    
def getLast(elem):
    
    return elem[-1]

def nearestNeighborOne(oneThird, twoThirds, k):
    
    classifiers = []
    for j in range(len(oneThird)):
        neighbors = findKNeighbors(twoThirds, k, oneThird[j])
        classifiers.append(neighbors[0][2])
    
    return classifiers
    
def nearestNeighborTwo(oneThird, twoThirds, k):  

    classifiers = []
    for j in range(len(oneThird)):
        classifier = []
        neighbors = findKNeighbors(twoThirds, k, oneThird[j])
        classifier.append(neighbors[0][2])
        classifier.append(neighbors[1][2])
        classifiers.append(classifier[np.random.binomial(1, 0.5)])

    return classifiers

  def getData(filename):
    
    file = open(filename)
    data = pd.read_csv(file, names=['sepal length','sepal width','petal length','petal width','target'])
    features = ['sepal length','sepal width','petal length','petal width']
    x = data.loc[:, features].values
    y = data.loc[:, 'target'].values    
    file.close()
    
    return x,y

def getPCA(x, y):
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    finalDataFrame = np.concatenate([principalComponents, y.reshape(150,1)], axis = 1)
    
    return finalDataFrame
  
def chooseCols(data):

    newData = data
    np.random.shuffle(newData)
    twoThirds = []
    oneThird = []
    for i in range(len(data)):
        if i < 100:
            twoThirds.append(data[i].tolist())
        elif i >= 100:
            oneThird.append(data[i].tolist())
            
    return twoThirds, oneThird
  
def findAccuracy(classifiers, oneThird):

    correct = 0
    for i in range(len(classifiers)):
        if classifiers[i] == oneThird[i][2]:
            correct += 1
            
    return (correct/len(oneThird))*100

def main():

    k = #I changed k here. It was either 1 or 2 depending on which function I wanted to use.
    x, y = getData('iris.data')
    finalDataFrame = getPCA(x, y)
    twoThirds, oneThird = chooseCols(finalDataFrame)
        
    if k == 1:
        sum = 0
        for i in range(100):
            sum += findAccuracy(nearestNeighborOne(oneThird, twoThirds, k), oneThird)
        print('Accuracy of 1-Nearest Neighbor Algorithm: ' + str(sum/100) +'%')
    if k == 2:
        sum = 0
        for i in range(100):
            sum += findAccuracy(nearestNeighborTwo(oneThird, twoThirds, k), oneThird)
        print('Accuracy of 2-Nearest Neighbor Algorithm: ' + str(sum/100) +'%')

main()
