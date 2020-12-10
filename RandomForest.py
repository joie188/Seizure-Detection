import numpy as np
import pandas as pd
import math
import random
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

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
        self.neg_size = np.count_nonzero(self.trainY == 0)
        self.trees = []
        self.features_used = []
        self.feature_decrease_gini = {}

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
    
    def build_tree(self, data_indices, features, weights, gini):
        best_feature, best_val, best_left, best_right, best_gini = self.best_split(data_indices, features, weights, gini)
        if best_feature is None:
            return None
        root = Node(best_feature, best_val, np.count_nonzero(self.trainY[data_indices] == 1), len(data_indices) - np.count_nonzero(self.trainY[data_indices] == 1))
        root.left = self.build_tree(best_left, features, weights, best_gini)
        root.right = self.build_tree(best_right, features, weights, best_gini)
        return root

    def train(self):
        self.trees = []
        for _ in range(self.num_trees):
            #randomly sample from X with replacement
            samples_indices = np.random.choice(self.trainX.shape[0], size=self.samples_per_tree, replace=True)

            #randomly choose features to consider
            features = np.random.choice(self.features_to_train_on, size=self.features_per_tree, replace=False)
            self.features_used.append(features)
            positive_class_count = np.count_nonzero(self.trainY[samples_indices] == 1)
            negative_class_count = self.samples_per_tree - positive_class_count

            initial_gini = self.gini_score([samples_indices], (1.0 / positive_class_count, 1.0 / negative_class_count))

            tree = self.build_tree(samples_indices, features, (1.0 / positive_class_count, 1.0 / negative_class_count), initial_gini)
            
            self.trees.append(tree)

    def predict(self, testX):
        '''
        label of 1 = seizure, label of 0 = no seizure
        '''
        predictions = np.zeros(testX.shape[0])
        for i in range(testX.shape[0]):
            predict_proba = self.predict_proba_ensemble(testX[i])
            if predict_proba[0] > predict_proba[1]:
                predictions[i] = 1
            else:
                predictions[i] = 0
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
        #seizure_weight = 1.0 / tree.num_samples_seizure
        #noseizure_weight = 1.0 / tree.num_samples_noseizure
        while tree is not None:
            if x[tree.feature] <= tree.value:
                if tree.left is None:
                    return (float(tree.num_samples_seizure) / (tree.num_samples_seizure + tree.num_samples_noseizure), float(tree.num_samples_noseizure) / (tree.num_samples_seizure + tree.num_samples_noseizure))
                tree = tree.left
            else:
                if tree.right is None:
                    return (float(tree.num_samples_seizure) / (tree.num_samples_seizure + tree.num_samples_noseizure), float(tree.num_samples_noseizure) / (tree.num_samples_seizure + tree.num_samples_noseizure))
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
    
    def best_split(self, data, features, weights, prev_gini):
        best_score = 2**31-1
        best_feature, best_val, best_left, best_right = None, None, None, None
        for index in features:
            feature, val, left, right, score = self.find_split(data, index, weights)
            if score < best_score:
                best_score = score
                best_feature, best_val, best_left, best_right = feature, val, left, right
        if best_feature in self.feature_decrease_gini:
            self.feature_decrease_gini[best_feature] += (prev_gini - best_score)
        else:
            self.feature_decrease_gini[best_feature] = (prev_gini - best_score)
        return best_feature, best_val, best_left, best_right, best_score

    def find_split(self, data, feature, weights):
        best_score = 2**31-1
        left = None
        right = None
        val = None
        for row_index in data:
            split_val = self.trainX[row_index][feature]
            left_temp, right_temp = self.perform_split(data, feature, split_val)
            if len(left_temp) < self.min_samples_leaf or len(right_temp) < self.min_samples_leaf:
                continue
            split_gini_score = self.gini_score([left_temp, right_temp], weights)
            if split_gini_score < best_score:
                best_score = split_gini_score
                left = left_temp
                right = right_temp
                val = split_val
        return feature, val, left, right, best_score

    def gini_score(self, splits, weights):
        '''
        splits is list of lists where each list has indices into trainX of rows in that group
        '''
        gini = 0
        all_samples = [index for split in splits for index in split]
        positive_class_weight = weights[0]
        negative_class_weight = weights[1]
        t_p = positive_class_weight * np.count_nonzero(self.trainY[all_samples] == 1) + negative_class_weight * np.count_nonzero(self.trainY[all_samples] == 0)
        for split in splits:
            if len(split) == 0:
                continue
            split_score = 0
            positive_class_count = np.count_nonzero(self.trainY[split] == 1) #seizure
            negative_class_count = np.count_nonzero(self.trainY[split] == 0) 
            t_c = positive_class_weight * positive_class_count + negative_class_weight * negative_class_count
            split_score += (positive_class_weight * positive_class_count/t_c) ** 2
            split_score += (negative_class_weight * negative_class_count/t_c) ** 2
            gini += t_c/t_p * (1.0 - split_score) 
        return gini

    def score(self, testX, testY):
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score

        pred = self.predict(testX)
        print("Precision:", precision_score(testY, pred, average='binary'))
        print("Recall:", recall_score(testY, pred, average='binary'))
        return np.sum(pred == testY) / testX.shape[0]

if __name__ == "__main__": 
    train = pd.read_csv('data/train_data.csv').values
    X_train = train[:, :-1].copy()
    Y_train = train[:, -1].copy()
    Y_train[Y_train==-1] = 0

    validate = pd.read_csv('data/val_data.csv').values
    X_val = validate[:, :-1].copy()
    y_val = validate[:, -1].copy()
    y_val[y_val==-1] = 0

    test = pd.read_csv('data/test_data.csv')
    X_test = test.values[:, :-1].copy()
    y_test = test.values[:, -1].copy()
    y_test[y_test==-1] = 0

    classifier = RandomForest(100, 50, X_train, Y_train, samples_per_tree=500)
    classifier.train()
    flatten_features = [feature for feature_list in classifier.features_used for feature in feature_list]
    features_used_freq = Counter(flatten_features)
    feature_freq = {}
    for tree in classifier.trees:
        def traverse_tree(tree):
            if tree is not None:
                if tree.feature in feature_freq:
                    feature_freq[tree.feature] += 1
                else:
                    feature_freq[tree.feature] = 1
                traverse_tree(tree.left)
                traverse_tree(tree.right)
        traverse_tree(tree)
    feature_to_ratio = {}
    feature_to_gini_decrease = {}
    for key in feature_freq:
        feature_to_ratio[key] = float(feature_freq[key]) / features_used_freq[key]
        feature_to_gini_decrease[key] = classifier.feature_decrease_gini[key] / features_used_freq[key]
    print(dict(sorted(feature_to_ratio.items(), key=lambda item: item[1])))
    print(dict(sorted(feature_to_gini_decrease.items(), key=lambda item: item[1])))

    #4, 13, 5, 17, 11, 14, 12, 10, 1, 0

    #4, 13, 14, 7, 15, 12, 5, 10, 1, 11

    # num_trees = [10, 50, 100]
    # num_samples_leaf = [10, 50, 25]
    # classifier = RandomForest(10, 10, X_train, Y_train, 500)
    # print(classifier.features_per_tree)

    # for num_tree in [50]:
    #     for num_sample in [50]:
    #         #classifier = RandomForestClassifier(n_estimators=num_tree, min_samples_leaf=num_sample, max_samples=500,class_weight="balanced_subsample")
    #         #classifier.fit(X_train, Y_train)
    #         classifier = RandomForest(num_tree, num_sample, X_train, Y_train, samples_per_tree=500)
    #         classifier.train()
    #         print(classifier.score(X_train, Y_train))
    #         print(classifier.score(X_val, y_val))
    #         print(classifier.score(X_test, y_test))


