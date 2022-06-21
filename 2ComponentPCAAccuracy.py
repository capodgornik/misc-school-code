'''
February 2020
Math 496t
'''

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

def main():
    dataSource = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    data = pd.read_csv(dataSource, names=['type', 'alcohol', 'malic acid', 'ash', 'alcalinity of ash', 'magnesium', 'total phenols',
        'flavanoids', 'nonflavanoid phenols', 'proanthocyanins', 'color intensity', 'hue', 'OD280/OD315 of diluted wines', 'proline'])
    features = ['alcohol', 'malic acid', 'ash', 'alcalinity of ash', 'magnesium', 'total phenols',
        'flavanoids', 'nonflavanoid phenols', 'proanthocyanins', 'color intensity', 'hue', 'OD280/OD315 of diluted wines', 'proline',]
    x = data.loc[:, features].values
    y = data.loc[:, ['type']].values
    x = StandardScaler().fit_transform(x)

    principalComponents = PCA(n_components = 2).fit_transform(x)
    principalDataFrame = pd.DataFrame(data = principalComponents, columns = ['Principal Component 1', 'Principal Component 2'])
            
    k = 3
    kMeans = KMeans(n_clusters = k).fit(principalComponents)        
 
    finalDataFrame = pd.concat([principalDataFrame, data[['type']]], axis = 1)
    graph = plt.figure(figsize = (7,5))
    w = graph.add_subplot(1,1,1) 
    w.set_title('PCA (with Two Components)', fontsize = 15)    
    w.set_xlabel('Principal Component 1', fontsize = 10)
    w.set_ylabel('Principal Component 2', fontsize = 10)
    types = [1, 2, 3, 'Centroids']
    colors = ['blue', 'tab:green', 'blueviolet']

    correct = 0
    for i in range(len(principalComponents)):
        prediction1 = np.array(principalComponents[i])
        prediction1 = prediction1.reshape(-1, len(prediction1))
        prediction = kMeans.predict(prediction1)
        if prediction[0] == y[i][0]:
            correct += 1

    print('Accuracy = ' + str((correct/len(y)) * 100) + '%')
    
    for type, color in zip(types, colors):
        coordinates = finalDataFrame['type'] == type
        w.scatter(finalDataFrame.loc[coordinates, 'Principal Component 1'],
                   finalDataFrame.loc[coordinates, 'Principal Component 2'],
                   c = color, s = 50)
    plt.scatter(kMeans.cluster_centers_[:, 0], kMeans.cluster_centers_[:, 1], marker = '^', s = 100,
    c = 'yellow', edgecolor = 'black', label = 'centroids')
    w.legend(types)
    plt.show()
main()
