import numpy as np
import pandas as pd
import math
import random
from sklearn.ensemble import RandomForestClassifier

class Node:
    def __init__(self, feature, value, num_samples_seizure, num_samples_noseizure):
        self.feature = feature
        self.value = value
        self.num_samples_seizure = num_samples_seizure
        self.num_samples_noseizure = num_samples_noseizure
        self.left = None
        self.right = None

class RandomForest:
    def __init__(self, num_trees, min_samples_leaf, trainX, trainY, samples_per_tree=None, features_per_tree=None, features_to_train_on = None):
        '''
        samples_per_tree is the number of samples (with replacement) used to construct each tree
        features_per_tree is the number of features to consider when making splits in each tree
        features_to_train_on is a list of indices that correspond to features, if None: all features are used
        '''
        self.num_trees = num_trees
        self.min_samples_leaf = min_samples_leaf
        self.trainX = trainX
        self.trainY = trainY
        self.pos_size = np.count_nonzero(self.trainY == 1)
        self.neg_size = np.count_nonzero(self.trainY == -1)
        self.trees = []

        if samples_per_tree is None:
            self.samples_per_tree = trainX.shape[0]
        else:
            self.samples_per_tree = samples_per_tree

        if features_per_tree is None:
            self.features_per_tree = int(math.sqrt(trainX.shape[1]))
        else:
            self.features_per_tree = features_per_tree
        
        if features_to_train_on is None:
            self.features_to_train_on = np.arange(trainX.shape[1])
        else:
            self.features_to_train_on = features_to_train_on
    
    def build_tree(self, data_indices, features):
        best_feature, best_val, best_left, best_right = self.best_split(data_indices, features)
        if best_feature is None:
            return None
        root = Node(best_feature, best_val, np.count_nonzero(self.trainY[data_indices] == 1), len(data_indices) - np.count_nonzero(self.trainY[data_indices] == 1))
        root.left = self.build_tree(best_left, features)
        root.right = self.build_tree(best_right, features)
        return root

    def train(self):
        self.trees = []
        for _ in range(self.num_trees):
            #randomly sample from X with replacement
            samples_indices = np.random.choice(self.trainX.shape[0], size=self.samples_per_tree, replace=True)

            #randomly choose features to consider
            features = np.random.choice(self.features_to_train_on, size=self.features_per_tree, replace=False)

            tree = self.build_tree(samples_indices, features)

            self.trees.append(tree)

    def predict(self, testX):
        '''
        label of 1 = seizure, label of 0 = no seizure
        '''
        predictions = np.zeros(testX.shape[0])
        for i in range(testX.shape[0]):
            predict_proba = self.predict_proba_ensemble(testX[i])
            seizure_weight = 1 / float(self.pos_size)
            noseizure_weight = 1 / float(self.neg_size)
            if predict_proba[0] > predict_proba[1]:
                predictions[i] = 1
            else:
                predictions[i] = -1
        return predictions

    def predict_proba_ensemble(self, x):
        '''
        returns mean (prob seizure, prob no seizure)
        '''
        probs_seizure = 0
        probs_noseizure = 0
        for tree in self.trees:
            (prob_seizure, prob_noseizure) = self.predict_proba(x, tree)
            probs_seizure += prob_seizure
            probs_noseizure += prob_noseizure
        return (probs_seizure, probs_noseizure)

    def predict_proba(self, x, tree):
        '''
        x is a row of data
        returns (prob of seizure, prob of not seizure)
        '''
        seizure_weight = 1.0 / tree.num_samples_seizure
        noseizure_weight = 1.0 / tree.num_samples_noseizure
        while tree is not None:
            if x[tree.feature] <= tree.value:
                if tree.left is None:
                    return (float(tree.num_samples_seizure) * seizure_weight / (tree.num_samples_seizure + tree.num_samples_noseizure), float(tree.num_samples_noseizure) * noseizure_weight / (tree.num_samples_seizure + tree.num_samples_noseizure))
                tree = tree.left
            else:
                if tree.right is None:
                    return (float(tree.num_samples_seizure) * seizure_weight / (tree.num_samples_seizure + tree.num_samples_noseizure), float(tree.num_samples_noseizure) * noseizure_weight / (tree.num_samples_seizure + tree.num_samples_noseizure))
                tree = tree.right

    def perform_split(self, data, feature, split_val):
        left = []
        right = []
        for row_index in data:
            if self.trainX[row_index][feature] <= split_val:
                left.append(row_index)
            else:
                right.append(row_index)
        return left, right
    
    def best_split(self, data, features):
        best_score = 2**31-1
        best_feature, best_val, best_left, best_right = None, None, None, None
        for index in features:
            feature, val, left, right, score = self.find_split(data, index)
            if score < best_score:
                best_score = score
                best_feature, best_val, best_left, best_right = feature, val, left, right
        return best_feature, best_val, best_left, best_right

    def find_split(self, data, feature):
        best_score = 2**31-1
        left = None
        right = None
        val = None
        for row_index in data:
            split_val = self.trainX[row_index][feature]
            left_temp, right_temp = self.perform_split(data, feature, split_val)
            if len(left_temp) < self.min_samples_leaf or len(right_temp) < self.min_samples_leaf:
                continue
            split_gini_score = self.gini_score([left_temp, right_temp])
            if split_gini_score < best_score:
                best_score = split_gini_score
                left = left_temp
                right = right_temp
                val = split_val
        return feature, val, left, right, best_score

    def gini_score(self, splits):
        '''
        splits is list of lists where each list has indices into trainX of rows in that group
        '''
        gini = 0
        num_samples = sum([len(split) for split in splits])
        for split in splits:
            if len(split) == 0:
                continue
            split_score = 0
            positive_class_count = np.count_nonzero(self.trainY[split] == 1) #seizure
            negative_class_count = np.count_nonzero(self.trainY[split] == -1) 
            positive_class_weight = 1
            negative_class_weight = 1
            split_score += (positive_class_weight * positive_class_count/len(split)) ** 2
            split_score += (negative_class_weight * negative_class_count/len(split)) ** 2
            gini += float(len(split))/float(num_samples) * (1.0 - split_score) 
        return gini

    def score(self, testX, testY):
        return np.sum(self.predict(testX) == testY) / testX.shape[0]

if __name__ == "__main__": 
    train = pd.read_csv('data/train_data.csv').values
    X_train = train[:, :-1].copy()
    Y_train = train[:, -1].copy()

    validate = pd.read_csv('data/val_data.csv').values
    X_val = validate[:, :-1].copy()
    y_val = validate[:, -1].copy()

    num_trees = [10, 50, 100]
    num_samples_leaf = [10, 25, 50]

    for num_tree in [10]:
        for num_sample in num_samples_leaf:
            #classifier = RandomForestClassifier(n_estimators=num_tree, min_samples_leaf=num_sample, max_samples=1000)
            #classifier.fit(X_train, Y_train)
            classifier = RandomForest(num_tree, num_sample, X_train, Y_train, samples_per_tree=1000)
            classifier.train()
            print(num_tree, num_sample)
            print(classifier.score(X_train, Y_train))
            print(classifier.score(X_val, y_val))

    # def traverse_tree(tree):
    #     if tree is not None:
    #         print(tree.feature)
    #         traverse_tree(tree.left)
    #         traverse_tree(tree.right)

    # classifier = RandomForest(10, 50, X_train, Y_train, samples_per_tree=1000)
    # classifier.train()
    # for tree in classifier.trees:
    #     print('new tree')
    #     traverse_tree(tree)

    # for num_tree in [50, 100]:
    #     num_samples_leaf = [10, 25, 50]
    #     if num_tree == 50:
    #         num_samples_leaf = [50]
    #     for num_samp in num_samples_leaf:
    #         classifier = RandomForest(num_tree, num_samp, X_train, Y_train, samples_per_tree=1000)
    #         classifier.train()
    #         print(num_tree, num_samp)
    #         print("Train: ", classifier.score(X_train, Y_train))
    #         print("Val: ", classifier.score(X_val, y_val))
