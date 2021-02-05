import random
from collections import Counter

from decisiontree import DecisionTree

class RandomForest:
    def __init__(self, num_trees, max_samples, max_X):
        self.num_trees = num_trees
        self.max_X = max_X
        self.max_samples = max_samples
        self.trees = []

    def train(self, D, X, Y):
        for i in range(self.num_trees):
            x = random.sample(X, int(len(X) * self.max_X))
            d = D.sample(frac=self.max_samples)
            tree = DecisionTree()
            tree.train(d, x, Y)
            self.trees.append(tree)

    def predict(self, D):
        y = []
        for tree in self.trees:
            y.append(tree.predict(D))
        result = []
        for i in range(len(D)):
            #https://docs.python.org/3/library/collections.html
            c = Counter()
            for j in range(len(self.trees)):
                c[y[j][i]] += 1
            pred, _ = c.most_common()[0]
            result.append(pred)

        return result
