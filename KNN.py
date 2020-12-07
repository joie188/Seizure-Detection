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
        """
        Parameters:
            K is an int representing the number of closest neighbors to consider
            distance_metric is one of euclidean or manhattan
            trainX is d by n array
            trainY is 1 by n array
        """
        self.trainX = trainX
        self.trainY = trainY
        self.pos_size = np.count_nonzero(self.trainY == 1)
        self.neg_size = self.trainY.shape[1] - self.pos_size
        self.K = K
        self.metric = distance_metric
    
    def calc_distances(self, testX):
        """
        Parameters:
            testX is d by m array
        Returns:
            an m x n array D where D[i, j] is the distance between test sample i and train sample j
        """
        dists = np.zeros((1, self.trainX.shape[1]))
        for i in range(testX.shape[1]):
            row = self.metric(self.trainX, testX[:, i:i+1])
            for j in range(len(row)):
                if self.trainY[:,j] == 1:
                    row[j] = row[j] / self.pos_size
                else:
                    row[j] = row[j] / self.neg_size
            dists = np.vstack((dists, row))
        return dists[1:, :]
        
    def find_top_neighbor_labels(self, dists):
        """
        Parameters:
            dists is  m x n array D where D[i, j] is the distance between test sample i and train sample j
        Returns:
            an m x K array L where L[i, j] is the label of the jth closest neighbor to test sample i
            in case of ties, the neighbor which appears first in the training set is chosen
        """
        neighbors = np.argsort(dists)[:, :self.K]
        neighbor_labels = self.trainY.squeeze()[neighbors]
        return neighbor_labels

    def find_top_neighbor(self, dists):
        top_distances = np.sort(dists)[:, :self.K]
        neighbor_labels = self.trainY.squeeze()[np.argsort(dists)[:, :self.K]]
        return top_distances, neighbor_labels
    
    def predict(self, testX):
        """
        Parameters:
            testX is d by m array
        Returns:
            predicted is 1 x m array P where P[0, i] is the predicted label for test sample i 
        """
        # neighbor_labels = self.find_top_neighbor_labels(self.calc_distances(testX))
        # predicted = np.hstack([np.bincount(vals).argmax() for vals in neighbor_labels])
        # return predicted
        dist, labels = self.find_top_neighbor(self.calc_distances(testX))
        predicted = []
        for r in range(dist.shape[0]):
            pos_sum, neg_sum = 0, 0
            vals = dist[r, :]
            lbls = labels[r, :]
            for c in range(len(vals)):
                if lbls[c] == 1:
                    pos_sum += vals[c]
                else:
                    neg_sum += vals[c]
            predicted.append(1 if pos_sum > neg_sum else -1)
        return predicted

    
    def score(self, testX, testY):
        """
        Parameters:
            testX is d by m array of input data
            testY is 1 by m array of labels for the input data
        Returns:
            the accuracy of the KNN predictions across the test set
        """
        score = np.sum(self.predict(testX) == testY) / testY.shape[1]
        return score


if __name__ == "__main__": 
    distance = euclidean_distance
    k = 5

    # train = np.loadtxt('data/data_train.csv')
    # X_train = train[:, 0:2].copy()
    # y_train = train[:, 2].copy()