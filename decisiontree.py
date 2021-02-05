import numpy as np

from node import Node


class DecisionTree:
    def __init__(self):
        self.root = None

    def train(self, D, X, Y):
        self.root = self._do_train(D, X, Y)

    def predict(self, D):
        y = []
        for i in range(len(D)):
            y.append(self._do_predict(self.root, D.iloc[[i]]))
        return y

    def _cost(self, D, Y):
        e = 0
        for k in D[Y].unique():
            Pk = len(D[D[Y] == k])/len(D)
            e = e + Pk * np.log2(Pk)
        return -e

    def _gain(self, D, X, Y):
        max_gain = 0
        i = X[0]
        costD = self._cost(D, Y)
        for col in X:
            g = 0
            for v in D[col].unique():
                Dv = D[D[col] == v]
                g += (len(Dv)/len(D))*self._cost(Dv, Y)
            g = costD - g
            if g >= max_gain:
                max_gain = g
                i = col
        return i

    def _do_train(self, D, X, Y):
        assert not D.empty
        if len(X)==0:
            #https://stackoverflow.com/questions/48590268/pandas-get-the-most-frequent-values-of-a-column/48590361
            return Node(D[Y].mode().sample(1).iloc[0], True)
        if len(D[Y].unique()) == 1:
            return Node(D[Y].iloc[0], True)
        x = self._gain(D, X, Y)
        tree = Node(x)
        for v in D[x].unique():
            tree.add_child(self._do_train(D[D[x] == v], [x1 for x1 in X if x1!=x], Y), v)
        return tree

    def _do_predict(self, tree, D):
        if tree.leaf:
            return tree.nodeId
        for child, value in tree.children:
            if value == D[tree.nodeId].values:
                return self._do_predict(child, D)
