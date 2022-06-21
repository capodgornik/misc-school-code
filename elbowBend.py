'''
February 2020
Math 496t
Explanation: The bend occurs at around 3 clusters, which is how many types of 
wine were in the dataset. Therefore, in this instance, the elbow bend method 
is a good way to decide how many clusters to partition the data into. 
'''

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def main():
    dataSource = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    data = pd.read_csv(dataSource, names=['type','alcohol', 'malic acid', 'ash', 'alcalinity of ash', 'magnesium', 'total phenols',
        'flavanoids', 'nonflavanoid phenols', 'proanthocyanins', 'color intensity', 'hue', 'OD280/OD315 of diluted wines', 'proline'])
    features = ['alcohol', 'malic acid', 'ash', 'alcalinity of ash', 'magnesium', 'total phenols',
        'flavanoids', 'nonflavanoid phenols', 'proanthocyanins', 'color intensity', 'hue', 'OD280/OD315 of diluted wines', 'proline',]
    x = data.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    
    squaredDistancesSum = []
    for i in range(1, 14):
        kMeans = KMeans(n_clusters=i)
        kMeans.fit(x)
        squaredDistancesSum.append(kMeans.inertia_)

    plt.plot(range(1, 14), squaredDistancesSum, marker = '.')
    plt.title('Elbow Bend Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distances to Closest Cluster')
    plt.show()

main()
