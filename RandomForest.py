import numpy as np
import math
import random

class Node:
    def __init__(self, feature, value, num_samples_seizure, num_samples_noseizure):
        self.feature = feature
        self.value = value
        self.num_samples_seizure = num_samples_seizure
        self.num_samples_noseizure = num_samples_noseizure
        self.left = None
        self.right = None

class RandomForest:
    def __init__(self, num_trees, min_samples_leaf, trainX, trainY, samples_per_tree=None, features_per_tree=None, ):
        self.num_trees = num_trees
        self.min_samples_leaf = min_samples_leaf
        self.trainX = trainX
        self.trainY = trainY
        self.trees = []

        if samples_per_tree is None:
            self.samples_per_tree = trainX.shape[0]
        else:
            self.samples_per_tree = samples_per_tree

        if features_per_tree is None:
            self.features_per_tree = math.sqrt(trainX.shape[1])
        else:
            self.features_per_tree = features_per_tree
    
    def build_tree(self, data_indices, features):
        best_feature, best_val, best_left, best_right = self.best_split(data_indices, features)
        if best_feature is None:
            return None
        root = Node(best_feature, best_val, np.count_nonzero(self.trainY[data_indices] == 1), len(data_indices) - np.count_nonzero(self.trainY[data_indices] == 1))
        root.left = self.build_tree(best_left, features)
        root.right = self.build_tree(best_right, features)
        return root

    def train(self, sample_weights):
        self.trees = []
        for _ in range(self.num_trees):
            #randomly sample from X with replacement
            samples_indices = np.random.choice(self.trainX.shape[0], size=self.samples_per_tree, replace=True)

            #randomly choose features to consider
            features = np.random.choice(self.trainX.shape[1], size=self.features_per_tree, replace=False)

            tree = self.build_tree(samples_indices, features)

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
        probs_seizure = []
        probs_noseizure = []
        for tree in self.trees:
            (prob_seizure, prob_noseizure) = self.predict_proba(x, tree)
            probs_seizure.append(prob_seizure)
            probs_noseizure.append(prob_noseizure)
        return (np.mean(np.array(probs_seizure)), np.mean(np.array(probs_noseizure)))

    def predict_proba(self, x, tree):
        '''
        x is a row of data
        returns (prob of seizure, prob of not seizure)
        '''
        while tree is not None:
            if x[tree.feature] <= tree.value:
                if tree.left is None:
                    total_samples = tree.num_samples_seizure + tree.num_samples_noseizure
                    return (float(tree.num_samples_seizure) / float(total_samples), float(tree.num_samples_noseizure) / float(total_samples))
                tree = tree.left
            else:
                if tree.right is None:
                    total_samples = tree.num_samples_seizure + tree.num_samples_noseizure
                    return (float(tree.num_samples_seizure) / float(total_samples), float(tree.num_samples_noseizure) / float(total_samples))
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
            split_score += (positive_class_count/len(split)) ** 2
            split_score += (1.0 - positive_class_count/len(split)) ** 2
            gini += float(len(split))/float(num_samples) * (1.0 - split_score) 
        return gini