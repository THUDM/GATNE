import os
import sys
import logging
import math
import argparse
import subprocess
import tqdm
import time
import numpy as np
from numpy import random
import tensorflow as tf

from collections import defaultdict
from six import iteritems
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
import sklearn.preprocessing
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Vocab

# from walk import RWGraph
from walk_n import RWGraph
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', nargs='?', default='data/amazon_sss.edges',
                        help='Input graph path')
    
    parser.add_argument('--features', nargs='?', default=None,
                        help='Input node features (npy)')

    parser.add_argument('--method', type=int, default=0,
                        help='Method: 0 for MNE, 1 for our')

    parser.add_argument('--epoch', type=int, default=5,
                        help='Number of epoch. Default is 5.')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of batch_size. Default is 64.')

    parser.add_argument('--eval_type', type=str, default='all',
                        help='The edge type for evaluation.')

    parser.add_argument('--dimensions', type=int, default=200,
                        help='Number of dimensions. Default is 200.')

    parser.add_argument('--meta_dim', type=int, default=10,
                        help='Number of meta dimensions. Default is 10.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 20.')

    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers. Default is 4.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def get_dict_neighbourhood_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except Exception as e:
        print(e)


def get_dict_AUC(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        tmp_score = get_dict_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(1)
        prediction_list.append(tmp_score)

    for edge in false_edges:
        tmp_score = get_dict_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(0)
        prediction_list.append(tmp_score)

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-len(true_edges)]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)

def generate_pairs(all_walks, vocab):
    pairs = []
    skip_window = 2
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

    # print(all_walks)

    for walks in all_walks:
        for walk in walks:
            for word in walk:
                raw_vocab[word] += 1
    # print(raw_vocab)

    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))
        index2word.append(word)

    index2word.sort(key=lambda word: vocab[word].count, reverse=True)
    for i, word in enumerate(index2word):
        vocab[word].index = i
    
    # print(index2word)
    # print([(word, vocab[word].count, vocab[word].index) for word in vocab.keys()])

    return vocab, index2word

def get_batches(pairs, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size

    result = []
    for idx in range(n_batches):
        x, y, t = [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])
            y.append(pairs[index][1])
            t.append(pairs[index][2])
        result.append((np.array(x).astype(np.int32), np.array(y).reshape(-1, 1).astype(np.int32), np.array(t).astype(np.int32)))
    return result

def generate_walks(network_data):
    base_network = network_data['Base']
    # base_walker = RWGraph(get_G_from_edges(base_network), walk_length=args.walk_length, num_walks=args.num_walks, workers=8, temp_folder='./tmp')
    # base_walks = base_walker.simulate_walks()
    
    # base_walks = list(get_G_from_edges(base_network).edges)
    # base_walks.extend([(y, x) for (x, y) in base_walks])
    
    # base_walker = RWGraph(get_G_from_edges(base_network), False, 1, 1)
    # base_walker.preprocess_transition_probs()
    # base_walks = base_walker.simulate_walks(args.num_walks, args.walk_length)

    base_walker = RWGraph(get_G_from_edges(base_network))
    base_walks = base_walker.simulate_walks(args.num_walks, args.walk_length)

    all_walks = []
    for layer_id in network_data:
        if layer_id == 'Base':
            continue

        tmp_data = network_data[layer_id]
        # start to do the random walk on a layer
        # layer_walker = RWGraph(get_G_from_edges(tmp_data), walk_length=args.walk_length, num_walks=args.num_walks, workers=8, temp_folder='./tmp')
        # layer_walks = layer_walker.simulate_walks()
        
        # layer_walks = list(get_G_from_edges(tmp_data).edges)
        # layer_walks.extend([(y, x) for (x, y) in layer_walks])

        # layer_walker = RWGraph(get_G_from_edges(tmp_data), False, 1, 1)
        # layer_walker.preprocess_transition_probs()
        # layer_walks = layer_walker.simulate_walks(args.num_walks, args.walk_length)

        layer_walker = RWGraph(get_G_from_edges(tmp_data))
        layer_walks = layer_walker.simulate_walks(args.num_walks, args.walk_length)

        all_walks.append(layer_walks)

    print('finish generating the walks')

    return base_walks, all_walks

def train_model(network_data, feature_dic, log_name):
    base_walks, all_walks = generate_walks(network_data)

    vocab, index2word = generate_vocab([base_walks])

    train_pairs = generate_pairs(all_walks, vocab)

    edge_types = list(network_data.keys())

    num_nodes = len(index2word)
    edge_type_count = len(edge_types) - 1
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = 200 # Dimension of the embedding vector.
    embedding_u_size = 10
    num_sampled = 5 # Number of negative examples to sample.

    neighbor_samples = 10

    neighbors = [[] for _ in range(num_nodes * edge_type_count)]
    for r in range(edge_type_count):
        g = network_data[edge_types[r]]
        for (x, y) in g:
            ix = vocab[x].index
            iy = vocab[y].index
            neighbors[ix * edge_type_count + r].append(iy * edge_type_count + r)
            neighbors[iy * edge_type_count + r].append(ix * edge_type_count + r)
        for i in range(num_nodes):
            j = i * edge_type_count + r
            if len(neighbors[j]) == 0:
                neighbors[j] = [i] * neighbor_samples
            elif len(neighbors[j]) < neighbor_samples:
                neighbors[j].extend(list(np.random.choice(neighbors[j], size=neighbor_samples-len(neighbors[j]))))
            elif len(neighbors[j]) > neighbor_samples:
                neighbors[j] = list(np.random.choice(neighbors[j], size=neighbor_samples))

    graph = tf.Graph()

    if feature_dic:
        feature_dim = len(list(feature_dic.values())[0])
        print('feature dimension: ' + str(feature_dim))
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        for key, value in feature_dic.items():
            if key in index2word:
                features[index2word.index(key), :] = np.array(value)

    with graph.as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)

        if feature_dic:
            node_features = tf.Variable(features, name='node_features', trainable=False)
            feature_weights = tf.Variable(tf.truncated_normal([feature_dim, embedding_size], stddev=1.0))

        # Parameters to learn
        node_embeddings = tf.Variable(tf.random_uniform([num_nodes, embedding_size], -1.0, 1.0))
        node_type_embeddings = tf.Variable(tf.random_uniform([edge_type_count * num_nodes, embedding_u_size], -1.0, 1.0))
        trans_weights = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_weights = tf.Variable(tf.truncated_normal([num_nodes, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([num_nodes]))

        node_neighbors = tf.Variable(neighbors, trainable=False)

        # Input data and re-orgenize size.
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None, 1])
        train_types = tf.placeholder(tf.int32, shape=[None])
        
        # Look up embeddings for words.
        node_embed = tf.nn.embedding_lookup(node_embeddings, train_inputs)
        trans_w = tf.nn.embedding_lookup(trans_weights, train_types)
        # Compute the softmax loss, using a sample of the negative labels each time.
        # loss_node2vec = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases,node_embed, train_labels, num_sampled, num_nodes))

        if args.method == 0:
            node_type_embed = tf.reshape(tf.nn.embedding_lookup(node_type_embeddings, train_inputs + train_types * num_nodes), [-1, 1, embedding_u_size])
        else:
            node_neigh = tf.nn.embedding_lookup(node_neighbors, train_inputs + train_types * num_nodes)

            node_embed_neighbors = tf.nn.embedding_lookup(node_type_embeddings, node_neigh)
            node_type_embed = tf.reshape(tf.reduce_mean(node_embed_neighbors, axis=1), [-1, 1, embedding_u_size])

        def my_norm(x):
            # return tf.nn.l2_normalize(x, axis=2)
            # return x / (tf.norm(x) / 100.0)
            # if tf.greater(tf.norm(x), tf.constant(1000)):
                # return x / (tf.norm(x) / 100.0)
            return x

        if feature_dic:
            node_feat = tf.nn.embedding_lookup(node_features, train_inputs)
            node_embed = node_embed + tf.matmul(node_feat, my_norm(feature_weights))

        # node_embed = tf.nn.l2_normalize(node_embed + tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size]), axis = 1)
        # node_embed = tf.nn.l2_normalize(tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size]), axis = 1)
        node_embed = node_embed + tf.reshape(tf.matmul(node_type_embed, my_norm(trans_w)), [-1, embedding_size])

        tmp_w = [tf.nn.embedding_lookup(trans_weights, i) for i in range(edge_type_count)]
        tmp_w = [tf.cond(tf.norm(tmp_w[i]) < tf.constant(1000.0), lambda: tmp_w[i], lambda: tmp_w[i] / (tf.norm(tmp_w[i]) / 1000.0)) for i in range(edge_type_count)]
        update = tf.assign(trans_weights, tmp_w)

        last_node_embed = node_embed
        # last_node_embed = tf.nn.l2_normalize(node_embed, axis=1)

        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=last_node_embed,
                num_sampled=num_sampled,
                num_classes=num_nodes))

        plot_loss = tf.summary.scalar("loss", loss)

        # Optimizer.
        optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

        # Add ops to save and restore all the variables.
        # saver = tf.train.Saver(max_to_keep=20)

        # Initializing the variables
        init = tf.global_variables_initializer()

    # Launch the graph
    print("Optimizing")
    last_score = 0
    with tf.Session(graph=graph) as sess:
        log_dir = "./log/" 
        writer = tf.summary.FileWriter("./runs/" + log_name, sess.graph) # tensorboard --logdir=./runs
        sess.run(init)

        print('Training')
        iter = 0
        for epoch in range(epochs):
            random.shuffle(train_pairs)
            batches = get_batches(train_pairs, batch_size)

            data_iter = tqdm.tqdm(enumerate(batches),
                                desc="EP:%d" % (epoch),
                                total=len(batches),
                                bar_format="{l_bar}{r_bar}")
            avg_loss = 0.0

            for i, data in data_iter:
                feed_dict = {train_inputs: data[0], train_labels: data[1], train_types: data[2]}
                _, loss_value, summary_str = sess.run([optimizer, loss, plot_loss], feed_dict)
                writer.add_summary(summary_str, iter)
                iter += 1

                avg_loss += loss_value

                if i % 5000 == 0:
                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "avg_loss": avg_loss / (i + 1),
                        "loss": loss_value
                    }
                    data_iter.write(str(post_fix))

                sess.run(update)
            
            final_model = dict(zip(edge_types[:-1], [dict() for _ in range(edge_type_count)]))
            for i in range(edge_type_count):
                for j in range(num_nodes):
                    final_model[edge_types[i]][index2word[j]] = np.array(sess.run(last_node_embed, {train_inputs: [j], train_types: [i]})[0])
            valid_aucs, valid_f1s, valid_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            for i in range(edge_type_count):
                if args.eval_type == 'all' or edge_types[i] in args.eval_type.split(','):
                    tmp_auc, tmp_f1, tmp_pr = get_dict_AUC(final_model[edge_types[i]], valid_true_data_by_edge[edge_types[i]], valid_false_data_by_edge[edge_types[i]])
                    valid_aucs.append(tmp_auc)
                    valid_f1s.append(tmp_f1)
                    valid_prs.append(tmp_pr)

                    tmp_auc, tmp_f1, tmp_pr = get_dict_AUC(final_model[edge_types[i]], testing_true_data_by_edge[edge_types[i]], testing_false_data_by_edge[edge_types[i]])
                    test_aucs.append(tmp_auc)
                    test_f1s.append(tmp_f1)
                    test_prs.append(tmp_pr)
            print('valid auc:', np.mean(valid_aucs))
            print('valid pr', np.mean(valid_prs))
            print('valid f1:', np.mean(valid_f1s))

            average_auc = np.mean(test_aucs)
            average_f1 = np.mean(test_f1s)
            average_pr = np.mean(test_prs)
            print('test auc:', average_auc)
            print('test pr:', average_pr)
            print('test f1:', average_f1)

            cur_score = np.mean(valid_aucs)
            if cur_score < last_score and epoch >= 4:
                break
            last_score = cur_score
    
    return average_auc, average_f1, average_pr


def train_model_new(network_data, feature_dic, log_name):
    base_walks, all_walks = generate_walks(network_data)

    vocab, index2word = generate_vocab([base_walks])

    train_pairs = generate_pairs(all_walks, vocab)

    edge_types = list(network_data.keys())

    num_nodes = len(index2word)
    edge_type_count = len(edge_types) - 1
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = 200 # Dimension of the embedding vector.
    feature_size = 20
    embedding_u_size = args.meta_dim
    u_num = edge_type_count # 2
    num_sampled = 5 # Number of negative examples to sample.
    dim_a = 20
    att_head = 1
    neighbor_samples = 10

    neighbors = [[[] for __ in range(edge_type_count)] for _ in range(num_nodes)]
    for r in range(edge_type_count):
        g = network_data[edge_types[r]]
        for (x, y) in g:
            ix = vocab[x].index
            iy = vocab[y].index
            neighbors[ix][r].append(iy)
            neighbors[iy][r].append(ix)
        for i in range(num_nodes):
            if len(neighbors[i][r]) == 0:
                neighbors[i][r] = [i] * neighbor_samples
            elif len(neighbors[i][r]) < neighbor_samples:
                neighbors[i][r].extend(list(np.random.choice(neighbors[i][r], size=neighbor_samples-len(neighbors[i][r]))))
            elif len(neighbors[i][r]) > neighbor_samples:
                neighbors[i][r] = list(np.random.choice(neighbors[i][r], size=neighbor_samples))

    # neighbors = [[[_] for __ in range(edge_type_count)] for _ in range(num_nodes)]

    graph = tf.Graph()

    if feature_dic:
        feature_dim = len(list(feature_dic.values())[0])
        print('feature dimension: ' + str(feature_dim))
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        for key, value in feature_dic.items():
            if key in index2word:
                features[index2word.index(key), :] = np.array(value)
        
    with graph.as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)

        if feature_dic:
            node_features = tf.Variable(features, name='node_features', trainable=False)
            feature_weights = tf.Variable(tf.truncated_normal([feature_dim, embedding_size], stddev=1.0)) # embedding_size
            linear = tf.layers.Dense(units=embedding_size, activation=tf.nn.tanh, use_bias=True)

            embed_trans = tf.Variable(tf.truncated_normal([feature_dim, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            u_embed_trans = tf.Variable(tf.truncated_normal([edge_type_count, feature_dim, embedding_u_size], stddev=1.0 / math.sqrt(embedding_size)))

        # Parameters to learn
        node_embeddings = tf.Variable(tf.random_uniform([num_nodes, embedding_size], -1.0, 1.0))
        node_type_embeddings = tf.Variable(tf.random_uniform([num_nodes, u_num, embedding_u_size], -1.0, 1.0))
        trans_weights = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, embedding_size // att_head], stddev=1.0 / math.sqrt(embedding_size)))
        # trans_weights_o = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        trans_weights_s1 = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, dim_a], stddev=1.0 / math.sqrt(embedding_size)))
        trans_weights_s2 = tf.Variable(tf.truncated_normal([edge_type_count, dim_a, att_head], stddev=1.0 / math.sqrt(embedding_size)))
        nce_weights = tf.Variable(tf.truncated_normal([num_nodes, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([num_nodes]))

        node_neighbors = tf.Variable(neighbors, trainable=False)

        # Input data and re-orgenize size.
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        # train_zero_inputs = tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None, 1])
        train_types = tf.placeholder(tf.int32, shape=[None])
        
        # Look up embeddings for words.
        if feature_dic:
            node_embed = tf.nn.embedding_lookup(node_features, train_inputs)
            node_embed = tf.matmul(node_embed, embed_trans)
        else:
            node_embed = tf.nn.embedding_lookup(node_embeddings, train_inputs)
        # node_embed_label = tf.nn.embedding_lookup(node_embeddings, tf.reshape(train_labels, [-1]))
        # node_type_embed = tf.nn.embedding_lookup(node_type_embeddings, train_inputs)
        # node_type_embed_o = tf.nn.embedding_lookup(tf.reshape(node_type_embeddings, [-1, embedding_u_size]), train_inputs * u_num + train_types)
        # node_type_embed_label = tf.nn.embedding_lookup(node_type_embeddings, tf.reshape(train_labels, [-1]))
        
        if args.method == 1 or args.method == 4:
            node_neigh = tf.nn.embedding_lookup(node_neighbors, train_inputs)
            if feature_dic:
                node_embed_neighbors = tf.nn.embedding_lookup(node_features, node_neigh)
                node_embed_tmp = tf.concat([tf.matmul(tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, 0], [-1, 1, -1, -1]), [-1, feature_dim]), tf.reshape(tf.slice(u_embed_trans, [i, 0, 0], [1, -1, -1]), [feature_dim, embedding_u_size])) for i in range(edge_type_count)], axis=0)
                # node_embed_tmp = tf.matmul(tf.reshape(node_embed_neighbors, []), u_embed_trans)
                node_type_embed = tf.transpose(tf.reduce_mean(tf.reshape(node_embed_tmp, [edge_type_count, -1, neighbor_samples, embedding_u_size]), axis=2), perm=[1,0,2])
            else:
                node_embed_neighbors = tf.nn.embedding_lookup(node_type_embeddings, node_neigh)
                node_embed_tmp = tf.concat([tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, i, 0], [-1, 1, -1, 1, -1]), [1, -1, neighbor_samples, embedding_u_size]) for i in range(edge_type_count)], axis=0)
                node_type_embed = tf.transpose(tf.reduce_mean(node_embed_tmp, axis=2), perm=[1,0,2])

                if args.method == 4:
                    node_type_embed = tf.nn.embedding_lookup(tf.reshape(node_type_embed, [-1, embedding_u_size]), train_types + tf.range(tf.shape(train_types)[0]) * edge_type_count)
        elif args.method == 3:
            node_type_embed = tf.nn.embedding_lookup(node_type_embeddings, train_inputs)
            
        # node_type_embed = tf.nn.embedding_lookup(node_type_embeddings, train_inputs)

        trans_w = tf.nn.embedding_lookup(trans_weights, train_types)
        trans_w_s1 = tf.nn.embedding_lookup(trans_weights_s1, train_types)
        trans_w_s2 = tf.nn.embedding_lookup(trans_weights_s2, train_types)
        # trans_w_o = tf.nn.embedding_lookup(trans_weights_o, train_types)
        
        if args.method != 4:
            attention = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.tanh(tf.matmul(node_type_embed, trans_w_s1)), trans_w_s2), [-1, u_num])), [-1, att_head, u_num])
            # attention = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(node_type_embed, trans_w_s1), [-1, u_num])), [-1, 1, u_num])
            node_type_embed = tf.matmul(attention, node_type_embed)
        else:
            node_type_embed = tf.reshape(node_type_embed, [-1, 1, embedding_u_size])

        # attention_label = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.tanh(tf.matmul(node_type_embed_label, trans_w_s1)), trans_w_s2), [-1, u_num])), [-1, att_head, u_num])
        # node_type_embed_label = tf.matmul(attention_label, node_type_embed_label)

        # node_embed + tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size])
        node_embed = node_embed + tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size])# + tf.reshape(tf.matmul(tf.reshape(node_type_embed_o, [-1, 1, embedding_u_size]), trans_w_o), [-1, embedding_size])
        # node_embed = tf.nn.l2_normalize(tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size]), axis = 1)

        # node_embed_label = tf.nn.l2_normalize(node_embed_label + tf.reshape(tf.matmul(node_type_embed_label, trans_w), [-1, embedding_size]))

        if feature_dic:
            node_feat = tf.nn.embedding_lookup(node_features, train_inputs)
            node_embed = node_embed + tf.matmul(node_feat, feature_weights)
            # node_embed = linear(tf.concat([node_embed, node_feat], axis=1))
            # node_embed = tf.matmul(node_feat, feature_weights)

        last_node_embed = tf.nn.l2_normalize(node_embed, axis=1)

        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=last_node_embed,
                num_sampled=num_sampled,
                num_classes=num_nodes))
        plot_loss = tf.summary.scalar("loss", loss)

        # loss2 = -tf.reduce_mean(tf.reduce_sum(tf.multiply(last_node_embed, node_embed_label), axis=-1))
        # plot_loss2 = tf.summary.scalar("loss2", loss2)

        # plot_loss_all = tf.summary.scalar("loss_all", loss + loss2)

        # Optimizer.
        optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

        # Add ops to save and restore all the variables.
        # saver = tf.train.Saver(max_to_keep=20)

        merged = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

        # Initializing the variables
        init = tf.global_variables_initializer()

    # Launch the graph
    print("Optimizing")
    last_score = 0
    with tf.Session(graph=graph) as sess:
        log_dir = "./log/"
        writer = tf.summary.FileWriter("./runs/" + log_name, sess.graph) # tensorboard --logdir=./runs
        sess.run(init)

        print('Training')
        iter = 0
        for epoch in range(epochs):
            random.shuffle(train_pairs)
            batches = get_batches(train_pairs, batch_size)

            data_iter = tqdm.tqdm(enumerate(batches),
                                desc="EP:%d" % (epoch),
                                total=len(batches),
                                bar_format="{l_bar}{r_bar}")
            avg_loss = 0.0

            for i, data in data_iter:
                feed_dict = {train_inputs: data[0], train_labels: data[1], train_types: data[2]}
                _, loss_value, summary_str = sess.run([optimizer, loss, merged], feed_dict)
                writer.add_summary(summary_str, iter)
                # writer.add_summary(summary_str2, iter)
                # writer.add_summary(summary_str3, iter)

                iter += 1

                avg_loss += loss_value

                if i % 5000 == 0:
                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "avg_loss": avg_loss / (i + 1),
                        "loss": loss_value
                    }
                    data_iter.write(str(post_fix))
            
            final_model = dict(zip(edge_types[:-1], [dict() for _ in range(edge_type_count)]))
            for i in range(edge_type_count):
                for j in range(num_nodes):
                    final_model[edge_types[i]][index2word[j]] = np.array(sess.run(last_node_embed, {train_inputs: [j], train_types: [i]})[0])
            valid_aucs, valid_f1s, valid_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            for i in range(edge_type_count):
                if args.eval_type == 'all' or edge_types[i] in args.eval_type.split(','):
                    tmp_auc, tmp_f1, tmp_pr = get_dict_AUC(final_model[edge_types[i]], valid_true_data_by_edge[edge_types[i]], valid_false_data_by_edge[edge_types[i]])
                    valid_aucs.append(tmp_auc)
                    valid_f1s.append(tmp_f1)
                    valid_prs.append(tmp_pr)

                    tmp_auc, tmp_f1, tmp_pr = get_dict_AUC(final_model[edge_types[i]], testing_true_data_by_edge[edge_types[i]], testing_false_data_by_edge[edge_types[i]])
                    test_aucs.append(tmp_auc)
                    test_f1s.append(tmp_f1)
                    test_prs.append(tmp_pr)
            print('valid auc:', np.mean(valid_aucs))
            print('valid pr', np.mean(valid_prs))
            print('valid f1:', np.mean(valid_f1s))

            average_auc = np.mean(test_aucs)
            average_f1 = np.mean(test_f1s)
            average_pr = np.mean(test_prs)
            print('test auc:', average_auc)
            print('test pr:', average_pr)
            print('test f1:', average_f1)

            cur_score = np.mean(valid_aucs)
            # if cur_score < last_score and epoch >= 4:
            #     break
            last_score = cur_score
    
    return average_auc, average_f1, average_pr

   
if __name__ == "__main__":
    args = parse_args()
    file_name = args.input
    print(args)
    if args.features:
        feature_dic = {}
        with open(args.features, 'r') as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                items = line.strip().split()
                feature_dic[items[0]] = items[1:]
        # feature_dic = np.load(args.features)[()]
    else:
        feature_dic = None

    log_name = file_name.split('/')[-1].split('.')[0] + '_method:%d' % args.method + '_eval-type:%s' % args.eval_type + '_b:%d' % args.batch_size + '_e:%d' % args.epoch

    training_data_by_type = load_training_data(file_name.split('.')[0] + '/train.txt')
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(file_name.split('.')[0] + '/valid.txt')
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name.split('.')[0] + '/test.txt')

    if args.method == 0 or args.method == 2:
        average_auc, average_f1, average_pr = train_model(training_data_by_type, feature_dic, log_name + '_' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    else:
        average_auc, average_f1, average_pr = train_model_new(training_data_by_type, feature_dic, log_name + '_' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    print('Overall ROC-AUC:', average_auc)
    print('Overall PR-AUC', average_pr)
    print('Overall F1:', average_f1)
  