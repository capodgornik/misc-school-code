import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def main():  
    	
    dataSource = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    data = pd.read_csv(dataSource, names=['type', 'alcohol', 'malic acid', 'ash', 'alcalinity of ash', 'magnesium', 'total phenols', 'flavanoids', 'nonflavanoid phenols', 'proanthocyanins', 'color intensity', 'hue', 'OD280/OD315 of diluted wines', 'proline'])
    features = ['alcohol', 'malic acid', 'ash', 'alcalinity of ash', 'magnesium', 'total phenols',
        'flavanoids', 'nonflavanoid phenols', 'proanthocyanins', 'color intensity', 'hue', 'OD280/OD315 of diluted wines', 'proline',]
    x = data.loc[:, features].values
    y = data.loc[:, ['type']].values
    
    k = 3
    kMeans = KMeans(n_clusters = k).fit(x)
    
    correct = 0
    for i in range(len(x)):
        
        prediction1 = np.array(x[i])
        prediction1 = prediction1.reshape(-1, len(prediction1))
        prediction = kMeans.predict(prediction1)
        if prediction[0] == y[i][0]:
            correct += 1

    print('Accuracy = ' + str((correct/len(y)) * 100) + '%')

main()
