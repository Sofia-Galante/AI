class Node:
    def __init__(self, x, leaf=False):
        self.nodeId = x
        self.leaf = leaf
        self.children = []

    def add_child(self, y, v):
        self.children.append([y, v])
