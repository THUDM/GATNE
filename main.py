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
from sklearn.metrics import roc_auc_score
import sklearn.preprocessing
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Vocab

from walk import RandomWalk
import Random_walk
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

    parser.add_argument('--dimensions', type=int, default=200,
                        help='Number of dimensions. Default is 200.')

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
    except:
        return 2+random.random()


def get_dict_AUC(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    allpos = 0
    for edge in true_edges:
        tmp_score = get_dict_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(1)
        # prediction_list.append(tmp_score)
        # for the unseen pair, we randomly give a prediction
        if tmp_score > 2:
            if tmp_score > 2.5:
                prediction_list.append(0)
            else:
                prediction_list.append(0)
            # prediction_list.append(random.uniform(-1,1))
        else:
            prediction_list.append(tmp_score)
            allpos += 1
    allneg = 0
    for edge in false_edges:
        tmp_score = get_dict_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(0)
        # prediction_list.append(tmp_score)
        # for the unseen pair, we randomly give a prediction
        if tmp_score > 2:
            if tmp_score > 2.5:
                prediction_list.append(0)
            else:
                prediction_list.append(0)
        #    prediction_list.append(random.uniform(-1,1))
        else:
            prediction_list.append(tmp_score)
            allneg += 1

    sorted_pred=prediction_list[:]
    sorted_pred.sort()
    median=sum(sorted_pred)/(allpos+allneg+0.0)
    for i in range(len(prediction_list)):
        if prediction_list[i]==0: # for =0 things, we give it to median
            prediction_list[i]=median + random.uniform(-1e-4,1e-4)
    sorted_pred=prediction_list[:]
    sorted_pred.sort()
    threshold=sorted_pred[-allpos]
    correct, fcorrect = 0, 0
    for i in range(len(prediction_list)):
        if prediction_list[i]>threshold:
            if true_list[i]==1:
                correct+=1
            else:
                fcorrect+=1

    precision=correct/(allpos+0.0)
    recall=correct/(correct+fcorrect+0.0)
    f1=2*precision*recall/(precision+recall)

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    return roc_auc_score(y_true, y_scores),precision,recall,f1

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

def train_deepwalk_embedding(walks, iteration=None):
    if iteration is None:
        iteration = 10 # original iter 100
    model = Word2Vec(walks, size=200, window=5, min_count=0, sg=1, workers=5, iter=iteration)
    return model

def train_model(network_data, feature_dic, log_name):
    base_network = network_data['Base']
    # base_G = Random_walk.RWGraph(get_G_from_edges(base_network), False, 1, 1)
    # base_G.preprocess_transition_probs()
    # base_walks = base_G.simulate_walks(20, 10)
    base_walker = RandomWalk(get_G_from_edges(base_network), walk_length=10, num_walks=20, workers=4)
    base_walks = base_walker.simulate_walks()
    # all_walks = [base_walks]
    all_walks = []
    edge_types = list(network_data.keys())
    for layer_id in network_data:
        if layer_id == 'Base':
            continue

        tmp_data = network_data[layer_id]
        # start to do the random walk on a layer
        # layer_G = Random_walk.RWGraph(get_G_from_edges(tmp_data), False, 1, 1)
        # layer_G.preprocess_transition_probs()
        # layer_walks = layer_G.simulate_walks(20, 10)
        layer_walker = RandomWalk(get_G_from_edges(tmp_data), walk_length=10, num_walks=20, workers=4)
        layer_walks = layer_walker.simulate_walks()
        all_walks.append(layer_walks)


    print('finish generating the walks')

    vocab, index2word = generate_vocab([base_walks])

    # node2vec_model = train_deepwalk_embedding(all_walks[0])

    train_pairs_base = generate_pairs([base_walks], vocab)
    train_pairs = generate_pairs(all_walks, vocab)

    num_nodes = len(index2word)
    edge_type_count = len(all_walks)
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = 200 # Dimension of the embedding vector.
    embedding_u_size = 10
    num_sampled = 5 # Number of negative examples to sample.

    graph = tf.Graph()

    if feature_dic:
        feature_dim = len(list(feature_dic.values())[0])
        print('feature dimension: ' + str(feature_dim))
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        for key, value in feature_dic.items():
            if key in index2word:
                features[index2word.index(key), :] = np.array(value)

    with graph.as_default():
        global_step_base = tf.Variable(0, name='global_step_base', trainable=False)
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

        # Input data and re-orgenize size.
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None, 1])
        train_types = tf.placeholder(tf.int32, shape=[None])
        
        # Look up embeddings for words.
        node_embed = tf.nn.embedding_lookup(node_embeddings, train_inputs)
        node_type_embed = tf.reshape(tf.nn.embedding_lookup(node_type_embeddings, train_inputs + train_types * num_nodes), [-1, 1, embedding_u_size])
        trans_w = tf.nn.embedding_lookup(trans_weights, train_types)
        # Compute the softmax loss, using a sample of the negative labels each time.
        # loss_node2vec = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases,node_embed, train_labels, num_sampled, num_nodes))

        def my_norm(x):
            # return tf.nn.l2_normalize(x, axis=2)
            # return x / (tf.norm(x) / 100.0)
            # if tf.greater(tf.norm(x), tf.constant(1000)):
                # return x / (tf.norm(x) / 100.0)
            return x

        if feature_dic:
            node_feat = tf.nn.embedding_lookup(node_features, train_inputs)
            node_embed = node_embed + 0.5 * tf.matmul(node_feat, my_norm(feature_weights))

        node_embed_base = tf.nn.l2_normalize(node_embed, axis = 1)
        node_embed = tf.nn.l2_normalize(node_embed + 0.5 * tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size]), axis = 1)
        # node_embed = tf.nn.l2_normalize(tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size]), axis = 1)
        # node_embed_base = node_embed
        # node_embed = node_embed + 0.5 * tf.reshape(tf.matmul(node_type_embed, my_norm(trans_w)), [-1, embedding_size])

        tmp_w = [tf.nn.embedding_lookup(trans_weights, i) for i in range(edge_type_count)]
        tmp_w = [tf.cond(tf.norm(tmp_w[i]) < tf.constant(1000.0), lambda: tmp_w[i], lambda: tmp_w[i] / (tf.norm(tmp_w[i]) / 1000.0)) for i in range(edge_type_count)]
        # update = tf.assign(trans_weights, [tmp_w[i] / (tf.norm(tmp_w[i]) / 100.0) for i in range(edge_type_count)])
        update = tf.assign(trans_weights, tmp_w)

        loss_base = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=node_embed_base,
                num_sampled=num_sampled,
                num_classes=num_nodes))
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=node_embed,
                num_sampled=num_sampled,
                num_classes=num_nodes))

        plot_loss_base = tf.summary.scalar("loss_base", loss_base)
        plot_loss = tf.summary.scalar("loss", loss)

        # Optimizer.
        optimizer_base = tf.train.AdamOptimizer().minimize(loss_base, global_step=global_step_base)
        optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

        # Add ops to save and restore all the variables.
        # saver = tf.train.Saver(max_to_keep=20)

        # merged = tf.summary.merge_all()

        # Initializing the variables
        init = tf.global_variables_initializer()

    # Launch the graph
    print("Optimizing")
    with tf.Session(graph=graph) as sess:
        log_dir = "./log/" 
        writer = tf.summary.FileWriter("./runs/" + log_name, sess.graph) # tensorboard --logdir=./runs
        sess.run(init)

        # print(sess.run(update))
        # exit()
        print('Base Training')
        iter = 0
        for epoch in range(0):
            random.shuffle(train_pairs_base)
            batches = get_batches(train_pairs_base, batch_size)

            data_iter = tqdm.tqdm(enumerate(batches),
                                desc="EP:%d" % (epoch),
                                total=len(batches),
                                bar_format="{l_bar}{r_bar}")
            avg_loss = 0.0

            for i, data in data_iter:
                feed_dict = {train_inputs: data[0], train_labels: data[1], train_types: data[2]}
                _, loss_value, summary_str = sess.run([optimizer_base, loss_base, plot_loss_base], feed_dict)
                writer.add_summary(summary_str, iter)
                iter += 1

                avg_loss += loss_value

                if i % 1000 == 0:
                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "avg_loss": avg_loss / (i + 1),
                        "loss": loss_value
                    }
                    data_iter.write(str(post_fix))

        print('Training for Each Type')
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

                if i % 1000 == 0:
                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "avg_loss": avg_loss / (i + 1),
                        "loss": loss_value
                    }
                    data_iter.write(str(post_fix))

                # sess.run(update)

        final_node_embeddings, final_type_embeddings, final_trans_weights = sess.run([node_embeddings, node_type_embeddings, trans_weights])
        if feature_dic:
            final_feature_weights = sess.run(feature_weights)


    final_model = dict()
    # print(final_node_embeddings)
    final_model['base'] = final_node_embeddings
    final_model['index2word'] = index2word
    if feature_dic:
        final_model['fea_tran'] = final_feature_weights
        # final_model['fea_tran'] = sklearn.preprocessing.normalize(final_feature_weights, axis=0)

    for edge_type in range(edge_type_count):
        final_model[edge_types[edge_type]] = final_node_embeddings + 0.5 * np.matmul(final_type_embeddings[edge_type * num_nodes: (edge_type + 1) * num_nodes,:], final_trans_weights[edge_type])
        # final_model[edge_types[edge_type]] = final_node_embeddings + 0.5 * np.matmul(final_type_embeddings[edge_type * num_nodes: (edge_type + 1) * num_nodes,:], sklearn.preprocessing.normalize(final_trans_weights[edge_type], axis=1))

    return final_model

def train_model_new(network_data, feature_dic, log_name):
    base_network = network_data['Base']
    # base_G = Random_walk.RWGraph(get_G_from_edges(base_network), False, 1, 1)
    # base_G.preprocess_transition_probs()
    # base_walks = base_G.simulate_walks(20, 10)
    base_walker = RandomWalk(get_G_from_edges(base_network), walk_length=10, num_walks=20, workers=4)
    base_walks = base_walker.simulate_walks()
    # all_walks = [base_walks]
    all_walks = []
    edge_types = list(network_data.keys())
    for layer_id in network_data:
        if layer_id == 'Base':
            continue

        tmp_data = network_data[layer_id]
        # start to do the random walk on a layer
        # layer_G = Random_walk.RWGraph(get_G_from_edges(tmp_data), False, 1, 1)
        # layer_G.preprocess_transition_probs()
        # layer_walks = layer_G.simulate_walks(20, 10)
        layer_walker = RandomWalk(get_G_from_edges(tmp_data), walk_length=10, num_walks=20, workers=4)
        layer_walks = layer_walker.simulate_walks()
        all_walks.append(layer_walks)


    print('finish generating the walks')

    vocab, index2word = generate_vocab([base_walks])

    # node2vec_model = train_deepwalk_embedding(all_walks[0])

    train_pairs_base = generate_pairs([base_walks], vocab)
    train_pairs = generate_pairs(all_walks, vocab)

    num_nodes = len(index2word)
    edge_type_count = len(all_walks)
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = 200 # Dimension of the embedding vector.
    embedding_u_size = 200
    u_num = 2 # edge_type_count
    num_sampled = 5 # Number of negative examples to sample.
    dim_a = 10
    att_head = 1

    graph = tf.Graph()

    if feature_dic:
        feature_dim = len(list(feature_dic.values())[0])
        print('feature dimension: ' + str(feature_dim))
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        for key, value in feature_dic.items():
            if key in index2word:
                features[index2word.index(key), :] = np.array(value)
        

    with graph.as_default():
        global_step_base = tf.Variable(0, name='global_step_base', trainable=False)
        global_step = tf.Variable(0, name='global_step', trainable=False)

        if feature_dic:
            node_features = tf.Variable(features, name='node_features', trainable=False)
            feature_weights = tf.Variable(tf.truncated_normal([feature_dim, embedding_size], stddev=1.0))

        # Parameters to learn
        node_embeddings = tf.Variable(tf.random_uniform([num_nodes, embedding_size], -1.0, 1.0))
        node_type_embeddings = tf.Variable(tf.random_uniform([num_nodes, u_num, embedding_u_size], -1.0, 1.0))
        trans_weights = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, embedding_size // att_head], stddev=1.0 / math.sqrt(embedding_size)))
        trans_weights_s1 = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, dim_a], stddev=1.0 / math.sqrt(embedding_size)))
        trans_weights_s2 = tf.Variable(tf.truncated_normal([edge_type_count, dim_a, att_head], stddev=1.0 / math.sqrt(embedding_size)))
        nce_weights = tf.Variable(tf.truncated_normal([num_nodes, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([num_nodes]))

        # Input data and re-orgenize size.
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        # train_zero_inputs = tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None, 1])
        train_types = tf.placeholder(tf.int32, shape=[None])
        
        # Look up embeddings for words.
        node_embed = tf.nn.embedding_lookup(node_embeddings, train_inputs)
        node_type_embed = tf.nn.embedding_lookup(node_type_embeddings, train_inputs)
        # node_type_embed = tf.nn.embedding_lookup(node_type_embeddings, train_zero_inputs)
        trans_w = tf.nn.embedding_lookup(trans_weights, train_types)
        trans_w_s1 = tf.nn.embedding_lookup(trans_weights_s1, train_types)
        trans_w_s2 = tf.nn.embedding_lookup(trans_weights_s2, train_types)

        # Compute the softmax loss, using a sample of the negative labels each time.
        # loss_node2vec = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases,node_embed, train_labels, num_sampled, num_nodes))

        if feature_dic:
            node_feat = tf.nn.embedding_lookup(node_features, train_inputs)
            node_embed = node_embed + 0.5 * tf.matmul(node_feat, feature_weights)

        attention = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.tanh(tf.matmul(node_type_embed, trans_w_s1)), trans_w_s2), [-1, u_num])), [-1, att_head, u_num])
        # attention = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(node_type_embed, trans_w_s1), [-1, u_num])), [-1, 1, u_num])
        node_type_embed = tf.matmul(attention, node_type_embed)

        node_embed_base = tf.nn.l2_normalize(node_embed, axis = 1)
        # node_embed = tf.nn.l2_normalize(node_embed + 0.5 * tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size]), axis = 1)
        node_embed = tf.nn.l2_normalize(tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size]), axis = 1)

        loss_base = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=node_embed_base,
                num_sampled=num_sampled,
                num_classes=num_nodes))
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=node_embed,
                num_sampled=num_sampled,
                num_classes=num_nodes))
        plot_loss_base = tf.summary.scalar("loss_base", loss_base)
        plot_loss = tf.summary.scalar("loss", loss)

        # Optimizer.
        optimizer_base = tf.train.AdamOptimizer().minimize(loss_base, global_step=global_step_base)
        optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

        # Add ops to save and restore all the variables.
        # saver = tf.train.Saver(max_to_keep=20)

        # merged = tf.summary.merge_all()

        # Initializing the variables
        init = tf.global_variables_initializer()

    # Launch the graph
    print("Optimizing")
    with tf.Session(graph=graph) as sess:
        log_dir = "./log/"
        writer = tf.summary.FileWriter("./runs/" + log_name, sess.graph) # tensorboard --logdir=./runs
        sess.run(init)

        # print(sess.run(node_embeddings))
        print('Base Training')
        iter = 0
        for epoch in range(0):
            random.shuffle(train_pairs_base)
            batches = get_batches(train_pairs_base, batch_size)

            data_iter = tqdm.tqdm(enumerate(batches),
                                desc="EP:%d" % (epoch),
                                total=len(batches),
                                bar_format="{l_bar}{r_bar}")
            avg_loss = 0.0

            for i, data in data_iter:
                feed_dict = {train_inputs: data[0], train_labels: data[1], train_types: data[2]}
                _, loss_value, summary_str = sess.run([optimizer_base, loss_base, plot_loss_base], feed_dict)
                writer.add_summary(summary_str, iter)
                iter += 1

                avg_loss += loss_value

                if i % 1000 == 0:
                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "avg_loss": avg_loss / (i + 1),
                        "loss": loss_value
                    }
                    data_iter.write(str(post_fix))

        print('Training for Each Type')
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

                if i % 1000 == 0:
                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "avg_loss": avg_loss / (i + 1),
                        "loss": loss_value
                    }
                    data_iter.write(str(post_fix))

        final_node_embeddings, final_type_embeddings = sess.run([node_embeddings, node_type_embeddings])
        # final_type_embeddings = [final_type_embeddings[0] for _ in range(final_type_embeddings.shape[0])]
        final_trans_weights, final_trans_weights_s1, final_trans_weights_s2 = sess.run([trans_weights, trans_weights_s1, trans_weights_s2])
        if feature_dic:
            final_feature_weights = sess.run(feature_weights)

    final_model = dict()
    # print(final_node_embeddings)
    final_model['base'] = final_node_embeddings
    final_model['index2word'] = index2word
    if feature_dic:
        final_model['fea_tran'] = final_feature_weights

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        e_sum = e_x.sum(axis=-1)
        e_sum = np.reshape(e_sum, [len(e_sum), 1])
        return e_x / e_sum

    # final_model['addition'] = {}
    # final_model['tran'] = {}
    for edge_type in range(edge_type_count):
        tmp = np.tanh(np.matmul(np.reshape(final_type_embeddings, [-1, embedding_u_size]), final_trans_weights_s1[edge_type,:,:]))
        # att = np.reshape(softmax(np.reshape(np.matmul(tmp, final_trans_weights_s2[edge_type,:,:]), [-1, edge_type_count])), [-1, 1, edge_type_count])
        att = np.reshape(softmax(np.reshape(np.matmul(tmp, final_trans_weights_s2[edge_type,:,:]), [-1, u_num])), [-1, att_head, u_num])

        # print(edge_type, att)

        # final_model['addition'][edge_types[edge_type]] = np.reshape(np.matmul(att, final_type_embeddings), [-1, att_head, embedding_u_size])
        # final_model['tran'][edge_types[edge_type]] = final_trans_weights[edge_type,:,:]

        addition = np.reshape(np.matmul(att, final_type_embeddings), [-1, embedding_u_size])

        # print(np.linalg.norm(final_node_embeddings))
        # print(np.linalg.norm(np.reshape(np.matmul(addition, final_trans_weights[edge_type]), [-1, embedding_size])))
        # print(np.dot(final_node_embeddings, np.reshape(np.matmul(addition, final_trans_weights[edge_type]), [-1, embedding_size])))

        final_model[edge_types[edge_type]] = final_node_embeddings + 0.5 * np.reshape(np.matmul(addition, final_trans_weights[edge_type]), [-1, embedding_size])
        # final_model[edge_types[edge_type]] = np.reshape(np.matmul(addition, final_trans_weights[edge_type]), [-1, embedding_size])
    return final_model

if __name__ == "__main__":
    args = parse_args()
    file_name = args.input
    print(args)
    if args.features:
        feature_dic = np.load(args.features)[()]
    else:
        feature_dic = None

    log_name = file_name.split('/')[-1].split('.')[0] + '_method:%d' % args.method + '_b:%d' % args.batch_size + '_e:%d' % args.epoch

    # In our experiment, we use 5-fold cross-validation
    number_of_groups = 5

    overall_MNE_performance = list()
    overall_MNE_f1 = list()

    for i in range(number_of_groups):
        training_data_by_type = load_training_data(file_name.split('.')[0] + '/train_%d.txt' % i)

        if args.method == 0:
            MNE_model = train_model(training_data_by_type, feature_dic, log_name + '_fold:' + str(i) + '_' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        else:
            MNE_model = train_model_new(training_data_by_type, feature_dic, log_name + '_fold:' + str(i) + '_' + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

        tmp_MNE_performance = 0
        tmp_MNE_f1 = 0
        testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name.split('.')[0] + '/test_%d.txt' % i)
        for edge_type in testing_true_data_by_edge:
            print('We are working on edge:', edge_type)
            print('number of training edges:', len(training_data_by_type[edge_type]))
            print('number of testing true edges:', len(testing_true_data_by_edge[edge_type]))
            print('number of testing false edges:', len(testing_false_data_by_edge[edge_type]))
           
            local_model = dict()
            for pos in range(len(MNE_model['index2word'])):
                # 0.5 is the weight parameter mentioned in the paper, which is used to show how important each relation type is and can be tuned based on the network.
                node = MNE_model['index2word'][pos]
                local_model[node] = MNE_model[edge_type][pos]
                if feature_dic:
                    local_model[node] = local_model[node] + 0.5 * np.dot(feature_dic[node], MNE_model['fea_tran'])
            tmp_MNE_score,precision,recall,f1 = get_dict_AUC(local_model, testing_true_data_by_edge[edge_type], testing_false_data_by_edge[edge_type])
            
            tmp_MNE_performance += tmp_MNE_score
            tmp_MNE_f1 += f1
            print('score:', tmp_MNE_score)
            print('precision,recall,f1:', precision,recall,f1)

        print('performance:', tmp_MNE_performance / (len(testing_true_data_by_edge)))
        print('f1:', tmp_MNE_f1 / (len(testing_true_data_by_edge)))
       
        overall_MNE_performance.append(tmp_MNE_performance / (len(testing_true_data_by_edge)))
        overall_MNE_f1.append(tmp_MNE_f1 / (len(testing_true_data_by_edge)))

    overall_MNE_performance = np.asarray(overall_MNE_performance)
    overall_MNE_f1 = np.asarray(overall_MNE_f1)
   
    print('Overall AUC:', overall_MNE_performance)
    print('Overall AUC:', np.mean(overall_MNE_performance))
    print('Overall std:', np.std(overall_MNE_performance))
    print('')

    print('Overall F1:', overall_MNE_f1)
    print('Overall F1:', np.mean(overall_MNE_f1))
    print('Overall std F1:', np.std(overall_MNE_f1))
    print('end')
