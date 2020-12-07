import math 
import numpy as np

def euclidean_distance(A, B):
    return np.sum([pow(a - b, 2) for (a, b) in zip(A, B)])**0.5

def chi_squared_distance(A, B):
    return 0.5 * np.sum([((a - b) ** 2) / (a + b)  
                      for (a, b) in zip(A, B)]) 

def city_block_distance(A, B):
    return np.sum([abs(a - b) for (a, b) in zip(A, B)]) 

class KNN:
    def __init__(self, K, distance_metric, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY
        self.K = K
        self.metric = distance_metric
    
    def calc_distances(self, testX):
        dists = np.vstack([self.metric(self.trainX, testX[:, i:i+1]) for i in range(testX.shape[1])])
        return dists
        
    def find_top_neighbor_labels(self, dists):
        neighbors = np.argsort(dists)[:, :self.K]
        neighbor_labels = self.trainY.squeeze()[neighbors]
        return neighbor_labels
    
    def predict(self, testX):
        neighbor_labels = self.find_top_neighbor_labels(self.calc_distances(testX))
        predicted = np.hstack([np.bincount(vals).argmax() for vals in neighbor_labels])
        return predicted
    
    def score(self, testX, testY):
        score = np.sum(self.predict(testX) == testY) / testY.shape[1]
        return score


if __name__ == "__main__": 
    distance = euclidean_distance
    k = 5

    train = np.loadtxt('data/data_train.csv')
    X_train = train[:, 0:2].copy()
    y_train = train[:, 2].copy()
    
    classifier = KNN(k, distance, X_train, y_train)