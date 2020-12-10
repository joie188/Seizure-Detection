import math 
import numpy as np
import pandas as pd

def euclidean_distance(A, B):
    return np.sqrt(np.sum((A - B)**2, axis=0))  

def chi_squared_distance(A, B):
    return 0.5* np.sum(((A-B)**2) / (A+B), axis=0)

def manhattan_distance(A, B):
    return np.sum(abs(A-B), axis=0) 

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
        self.neg_size = np.count_nonzero(self.trainY == -1)
        self.K = K
        self.metric = distance_metric
    
    # def calc_distances(self, testX):
    #     """
    #     Parameters:
    #         testX is d by m array
    #     Returns:
    #         an m x n array D where D[i, j] is the distance between test sample i and train sample j
    #     """
    #     dists = np.zeros((1, self.trainX.shape[1]))
    #     for i in range(testX.shape[1]):
    #         row = self.metric(self.trainX, testX[:, i:i+1])
    #         for j in range(len(row)):
    #             # scale by class of respective class
    #             if self.trainY[j] == 1:
    #                 row[j] = row[j] #/ self.pos_size
    #             else:
    #                 row[j] = row[j] #/ self.neg_size
    #         dists = np.vstack((dists, row))
    #     return dists[1:, :]

    def calc_distances(self, testX):
        dists = np.vstack([self.metric(self.trainX, testX[:, i:i+1]) for i in range(testX.shape[1])])
        return dists
        
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
        top_labels = self.trainY.squeeze()[np.argsort(dists)[:, :self.K]]
        return top_distances, top_labels
    
    def predict(self, testX):
        """
        Parameters:
            testX is d by m array
        Returns:
            predicted is 1 x m array P where P[0, i] is the predicted label for test sample i 
        """
        neighbor_labels = self.find_top_neighbor_labels(self.calc_distances(testX))
        neighbor_labels = neighbor_labels.astype(int)
        predicted = np.hstack([np.bincount(vals).argmax() for vals in neighbor_labels])
        return predicted
        # dist, labels = self.find_top_neighbor(self.calc_distances(testX))
        # predicted = []
        # for r in range(dist.shape[0]):
        #     pos_sum, neg_sum = 0, 0 #weighted sum of class 1, -1
        #     vals = dist[r, :]
        #     lbls = labels[r, :]
        #     for c in range(len(vals)):
        #         if lbls[c] == 1:
        #             pos_sum += vals[c]
        #         else:
        #             neg_sum += vals[c]
        #     predicted.append(1 if pos_sum > neg_sum else -1)
        # return predicted

    
    def score(self, testX, testY):
        """
        Parameters:
            testX is d by m array of input data
            testY is 1 by m array of labels for the input data
        Returns:
            the accuracy of the KNN predictions across the test set
        """
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score

        pred = self.predict(testX)
        print("Precision:", precision_score(testY, pred, average='binary'))
        print("Recall:", recall_score(testY, pred, average='binary'))
        score = np.sum(pred == testY) / len(testY)
        return score


if __name__ == "__main__": 
    #distance = chi_squared_distance
    #k = 3
    import imblearn
    from imblearn.over_sampling import SMOTE
    from sklearn.neighbors import KNeighborsClassifier
    import pandas as pd
    import numpy as np

    features = [14, 13, 4, 7, 12, 1, 15, 10] 

    train = pd.read_csv('data/train_data.csv').values
    X_train = train[:, features].copy()
    y_train = train[:, -1].copy()
    y_train[y_train==-1] = 0

    validate = pd.read_csv('data/val_data.csv').values
    X_val = validate[:, features].copy()
    y_val = validate[:, -1].copy()
    y_val[y_val==-1] = 0

    test = pd.read_csv('data/test_data.csv')
    X_test = test.values[:, features].copy()
    y_test = test.values[:, -1].copy()
    y_test[y_test==-1] = 0

    oversample = SMOTE()
    x_train, y_train = oversample.fit_resample(X_train, y_train)

    for k in [91]:
        for distance in [ chi_squared_distance]:
            classifier = KNN(k, distance, x_train.T, y_train)
            print("Test:",1-classifier.score(X_test.T, y_test))
            # print("K:", k, "Distance:", distance)
            # print("Training:", 1-classifier.score(x_train.T, y_train))
            # print("Validation:",1-classifier.score(X_val.T, y_val))
            print()