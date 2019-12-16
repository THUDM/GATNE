import random

import networkx as nx
import numpy as np


class RWGraph():
    def __init__(self, nx_G, node_type=None):
        self.G = nx_G
        self.node_type = node_type

    def walk(self, walk_length, start, schema=None):
        # Simulate a random walk starting from start node.
        G = self.G

        rand = random.Random()

        if schema:
            schema_items = schema.split('-')
            assert schema_items[0] == schema_items[-1]

        walk = [start]
        while len(walk) < walk_length:
            cur = walk[-1]
            candidates = []
            for node in G[cur].keys():
                if schema == None or self.node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]:
                    candidates.append(node)
            if candidates:
                walk.append(rand.choice(candidates))
            else:
                break
        return [str(node) for node in walk]

    def simulate_walks(self, num_walks, walk_length, schema=None):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        # print('Walk iteration:')
        if schema is not None:
            schema_list = schema.split(',')
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                if schema is None:
                    walks.append(self.walk(walk_length=walk_length, start=node))
                else:
                    for schema_iter in schema_list:
                        if schema_iter.split('-')[0] == self.node_type[node]:
                            walks.append(self.walk(walk_length=walk_length, start=node, schema=schema_iter))

        return walks
