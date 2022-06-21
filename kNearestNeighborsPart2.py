'''
April 2020
Math 496t
Explanation: I ran the code a number of times to see what would happen 
and included the results from four of my trials. Each time I ran my code, 
the accuracies for a given value of k were similar. For each trial, each 
accuracy was the average of the accuracies of running the k-Nearest Neighbor 
algorithm 100 times. The results of this code show that the accuracy of the 
k-Nearest Neighbor algorithm decreases as k gets larger. In fact, for the 
majority of the outputs, either the 1-Nearest Neighbor and 2-Nearest Neighbor 
algorithms were tied, or the 1-Nearest Neighbor algorithm was slightly better 
than the 2-Nearest Neighbor algorithm. The 2-Nearest Neighbor algorithm was 
usually better than the k-Nearest Neighbor algorithms when k > 2, and the 
1-Nearest Neighbor algorithm was usually better for k-Nearest Neighbor algorithms 
for k > 1 and always better than the k-Nearest Neighbor algorithms when k > 3. 
That the accuracy decreases as k increases makes sense. As k gets larger, the number 
of neighbors chosen increases to include points that might not be close to the point 
that is being labelled. Furthermore, as discussed in the mini-lecture, a k that 
includes every data point makes the position of the point being labelled irrelevant 
since the algorithm is basically just finding out what the majority of the data labels 
are for the labelled data and labelling the unlabelled point with that label. 

'''

import math 
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import random
import matplotlib.pyplot as plt

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
    
def nearestNeighborsK(trainingData, x, k):
    
    classifiers = []
    for j in range(len(x)):
        classifier = []
        classifierVotes = [['Iris-setosa', 0], ['Iris-versicolor', 0], ['Iris-virginica', 0]]
        neighbors = findKNeighbors(trainingData, k, x[j])
        for i in range(k):
            classifier.append(neighbors[i][2]) 
            if neighbors[i][2] == 'Iris-setosa':
                classifierVotes[0][1] += 1
            elif neighbors[i][2] == 'Iris-versicolor':
                classifierVotes[1][1] += 1
            elif neighbors[i][2] == 'Iris-virginica':
                classifierVotes[2][1] += 1                
        trueClassifier = breakTies(classifierVotes)
        classifiers.append(trueClassifier)
        
    return classifiers
  
def breakTies(votes):
    
    max = 0
    classifierPossibilities = []
    for i in range(len(votes)):
        if votes[i][1] >= max and votes[i][1] != 0:
            max = votes[i][1]
            classifierPossibilities.append(votes[i])
    if len(classifierPossibilities) == 1:
        return classifierPossibilities[0][0]
    elif len(classifierPossibilities) > 1:
        max2 = 0
        classifierPossibilities2 = []
        for j in range(len(classifierPossibilities)):
            if classifierPossibilities[j][1] > max2:
                max2 = classifierPossibilities[j][1]
                classifierPossibilities2.append(classifierPossibilities[j])
        if len(classifierPossibilities2) == 1:
            return classifierPossibilities2[0][0]
        elif len(classifierPossibilities2) > 1:
            randomNum = random.randint(0,len(classifierPossibilities2)-1)
            return classifierPossibilities2[randomNum][0]

def findAccuracy(classifiers, oneThird):

    correct = 0
    for i in range(len(classifiers)):
        if classifiers[i] == oneThird[i][2]:
            correct += 1
            
    return (correct/len(oneThird))*100

def graphAccuracyVsK(accuracies):
    
    plt.plot(range(1, 11), accuracies, marker = '.')
    plt.title('Accuracy vs. K')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()    
    
def main():

    x, y = getData('iris.data')
    finalDataFrame = getPCA(x, y)
    
    twoThirds, oneThird = chooseCols(finalDataFrame)
    
    accuracies = []
    for k in range(1,11):
        sum = 0
        for i in range(100):
            sum += findAccuracy(nearestNeighborsK(twoThirds, oneThird, k), oneThird)
        accuracies.append(sum/100)
        print('Accuracy of '+ str(k) + '-Nearest Neighbor Algorithm: ' + str(sum/100) +'%')
        
    graphAccuracyVsK(accuracies)
    
main()
