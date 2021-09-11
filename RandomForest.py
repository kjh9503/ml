import numpy as np
import random
from collections import Counter
from tqdm import tqdm

class RandomForest:
    def __init__(self, max_depth, min_samples_split, ratio, n_estimators, n_features):
        # train
        self.trees = []
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.ratio = ratio
        self.n_estimators = n_estimators
        self.n_features = n_features

    def try_split(self, dataset, index, value):

        left = []
        right = []
        for sample in dataset:
            if sample[index] < value:
                left.append(sample)
            else:
                right.append(sample)
        return left, right

    def label_decision(self, group):
        '''
        return the label with highest frequency
        '''
        labels = [sample[-1] for sample in group]
        return max(dict(Counter(labels)).items(), key=lambda x: x[1])[0]

    def subsample(self, dataset):
        '''
        simple random sampling with replacement
        '''
        sample = []
        n_sample = round(len(dataset) * self.ratio)
        while len(sample) < n_sample:
            index = np.random.randint(len(dataset))
            sample.append(dataset[index])
        return sample

    def gini_index(self, groups, class_values):
        gini = 0
        total_size = sum(len(i) for i in groups)
        for group in groups:
            size = len(group)
            proportion = size / total_size
            portion_sqrd_sum = 0
            if size == 0:
                continue
            for class_value in class_values:
                cnt = [sample[-1] for sample in group].count(class_value)
                portion_sqrd = (cnt / size) ** 2
                portion_sqrd_sum += portion_sqrd
            gini += proportion * (1 - portion_sqrd_sum)
        return gini

    def get_split(self, train_samples):
        # randomly selected n features
        features = []
        while True:
            index = np.random.randint(len(train_samples[0]) - 1)
            if index not in features:
                features.append(index)
            if len(features) == self.n_features:
                break

        # get set of uniq labels
        class_values = list(Counter(sample[-1] for sample in train_samples).keys())

        # init params
        best_score = 10000  # minimum Gini index
        best_feature_idx = None  # index of best column
        best_threshold = None  # best cut-off value of the best column
        best_groups = None  # best groups

        # loop through selected features to get the minimum Gini index
        for feature_index in features:
            for sample in train_samples:
                feature_value = sample[feature_index]
                groups = self.try_split(train_samples, feature_index, feature_value)
                # groups = (left, right)
                gini = self.gini_index(groups, class_values)
                if gini < best_score:
                    best_feature_idx = feature_index
                    best_threshold = feature_value
                    best_score = gini
                    best_groups = groups
        return {'feature': best_feature_idx, 'threshold': best_threshold, 'groups': best_groups}

    def split(self, node, cur_depth):
        # node = {'feature' : (int)best_feature_idx, 'threshold' : (int)best_threshold, 'groups' : (two list)groups}
        left, right = node['groups']
        del (node['groups'])
        # node = {'feature' : (int)best_feature_idx, 'threshold' : (int)best_threshold}

        # check for a no split
        if left == [] or right == []:
            node['left'] = node['right'] = self.label_decision(left + right)
            return

        # check for max depth
        if cur_depth >= self.max_depth:
            node['left'] = self.label_decision(left)
            node['right'] = self.label_decision(right)
            return

        # process left child
        if len(left) <= self.min_samples_split:
            node['left'] = self.label_decision(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], cur_depth + 1)

        # process right child
        if len(right) <= self.min_samples_split:
            node['right'] = self.label_decision(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], cur_depth + 1)

    def build_tree(self, train_samples):
        root = self.get_split(train_samples)
        # root = {'feature' : (int)best_feature_idx, 'threshold' : (int)best_threshold, 'groups' : (two_list)groups}
        self.split(root, 1)
        return root

    # train
    def fit(self, train_datasets):
        for i in tqdm(range(self.n_estimators)):
            train_samples = self.subsample(train_datasets)
            tree = self.build_tree(train_samples)
            self.trees.append(tree)

    def dt_predict(self, node, sample):
        # node = {'feature' : (int)best_feature_idx, 'threshold' : (int)best_threshold, 'left' : list, 'right' : right}

        if sample[node['feature']] < node['threshold']:
            if type(node['left']) == dict:
                return self.dt_predict(node['left'], sample)
            else:
                return node['left']
        else:
            if type(node['right']) == dict:
                return self.dt_predict(node['right'], sample)
            else:
                return node['right']

    def bagging_predict(self, sample):
        '''
        make a prediction with a list of bagged trees
        '''
        bagging_predictions = [self.dt_predict(tree, sample) for tree in self.trees]
        # 랜.포 안에 있는 tree들에 의한 major voting
        return max(dict(Counter(bagging_predictions)).items(), key=lambda x: x[1])[0]

    # predict
    def predict(self, test_datasets):
        predictions = [self.bagging_predict(sample) for sample in test_datasets]
        return predictions


def scoring(classifier, test_data, answer):
  y_pred = classifier.predict(test_data)
  cnt = 0
  for i in range(len(test_data)):
    if y_pred[i] == answer[i]:
      cnt += 1
  return cnt/len(test_data) * 100