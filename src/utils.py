import argparse
from collections import defaultdict

import networkx as nx
import numpy as np
from gensim.models.keyedvectors import Vocab
from six import iteritems
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)

from walk import RWGraph


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='data/amazon',
                        help='Input dataset path')
    
    parser.add_argument('--features', type=str, default=None,
                        help='Input node features')

    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch. Default is 100.')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='Number of batch_size. Default is 64.')

    parser.add_argument('--eval-type', type=str, default='all',
                        help='The edge type(s) for evaluation.')
    
    parser.add_argument('--schema', type=str, default=None,
                        help='The metapath schema (e.g., U-I-U,I-U-I).')

    parser.add_argument('--dimensions', type=int, default=200,
                        help='Number of dimensions. Default is 200.')

    parser.add_argument('--edge-dim', type=int, default=10,
                        help='Number of edge embedding dimensions. Default is 10.')
    
    parser.add_argument('--att-dim', type=int, default=20,
                        help='Number of attention dimensions. Default is 20.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 20.')

    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')
    
    parser.add_argument('--negative-samples', type=int, default=5,
                        help='Negative samples for optimization. Default is 5.')
    
    parser.add_argument('--neighbor-samples', type=int, default=10,
                        help='Neighbor samples for aggregation. Default is 10.')

    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience. Default is 5.')
    
    return parser.parse_args()

def get_G_from_edges(edges):
    edge_dict = dict()
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    tmp_G = nx.Graph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        x = edge_key.split('_')[0]
        y = edge_key.split('_')[1]
        tmp_G.add_edge(x, y)
        tmp_G[x][y]['weight'] = weight
    return tmp_G

def load_training_data(f_name):
    print('We are loading data from:', f_name)
    edge_data_by_type = dict()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    print('Total training nodes: ' + str(len(all_nodes)))
    return edge_data_by_type


def load_testing_data(f_name):
    print('We are loading data from:', f_name)
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            x, y = words[1], words[2]
            if int(words[3]) == 1:
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    return true_edge_data_by_type, false_edge_data_by_type

def load_node_type(f_name):
    print('We are loading node type from:', f_name)
    node_type = {}
    with open(f_name, 'r') as f:
        for line in f:
            items = line.strip().split()
            node_type[items[0]] = items[1]
    return node_type

def generate_walks(network_data, num_walks, walk_length, schema, file_name):
    if schema is not None:
        node_type = load_node_type(file_name + '/node_type.txt')
    else:
        node_type = None

    all_walks = []
    for layer_id in network_data:
        tmp_data = network_data[layer_id]
        # start to do the random walk on a layer

        layer_walker = RWGraph(get_G_from_edges(tmp_data))
        layer_walks = layer_walker.simulate_walks(num_walks, walk_length, schema=schema)

        all_walks.append(layer_walks)

    print('Finish generating the walks')

    return all_walks

def generate_pairs(all_walks, vocab, window_size):
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        for walk in walks:
            for i in range(len(walk)):
                for j in range(1, skip_window + 1):
                    if i - j >= 0:
                        pairs.append((vocab[walk[i]].index, vocab[walk[i - j]].index, layer_id))
                    if i + j < len(walk):
                        pairs.append((vocab[walk[i]].index, vocab[walk[i + j]].index, layer_id))
    return pairs

def generate_vocab(all_walks):
    index2word = []
    raw_vocab = defaultdict(int)

    for walks in all_walks:
        for walk in walks:
            for word in walk:
                raw_vocab[word] += 1

    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))
        index2word.append(word)

    index2word.sort(key=lambda word: vocab[word].count, reverse=True)
    for i, word in enumerate(index2word):
        vocab[word].index = i
    
    return vocab, index2word

def get_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except Exception as e:
        pass


def evaluate(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    true_num = 0
    for edge in true_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

    for edge in false_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)
