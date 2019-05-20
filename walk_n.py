import numpy as np
import networkx as nx
import random


class RWGraph():
    def __init__(self, nx_G, alpha=0.0):
        self.G = nx_G
        self.alpha = alpha

    def walk(self, walk_length, start):
        # Simulate a random walk starting from start node.
        G = self.G

        rand = random.Random()

        if start:
            walk = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            walk = [rand.choice(list(G.keys()))]

        while len(walk) < walk_length:
            cur = walk[-1]
            if len(G[cur]) > 0:
                if rand.random() >= self.alpha:
                    walk.append(rand.choice(list(G[cur].keys())))
                else:
                    walk.append(walk[0])
            else:
                break
        return [str(node) for node in walk]

    def simulate_walks(self, num_walks, walk_length):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        # print('Walk iteration:')
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.walk(walk_length=walk_length, start=node))

        return walks


