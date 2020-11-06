import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
from numpy import random

from utils import *


def get_batches(pairs, neighbors, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size

    for idx in range(n_batches):
        x, y, t, neigh = [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])
            y.append(pairs[index][1])
            t.append(pairs[index][2])
            neigh.append(neighbors[pairs[index][0]])
        yield (np.array(x).astype(np.int32), np.array(y).reshape(-1, 1).astype(np.int32), np.array(t).astype(np.int32), np.array(neigh).astype(np.int32)) 

def train_model(network_data, feature_dic, log_name):
    vocab, index2word, train_pairs = generate(network_data, args.num_walks, args.walk_length, args.schema, file_name, args.window_size, args.num_workers, args.walk_file)

    edge_types = list(network_data.keys())

    num_nodes = len(index2word)
    edge_type_count = len(edge_types)
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions # Dimension of the embedding vector.
    embedding_u_size = args.edge_dim
    u_num = edge_type_count
    num_sampled = args.negative_samples # Number of negative examples to sample.
    dim_a = args.att_dim
    att_head = 1
    neighbor_samples = args.neighbor_samples 

    neighbors = generate_neighbors(network_data, vocab, num_nodes, edge_types, neighbor_samples)

    graph = tf.Graph()

    if feature_dic is not None:
        feature_dim = len(list(feature_dic.values())[0])
        print('feature dimension: ' + str(feature_dim))
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        for key, value in feature_dic.items():
            if key in vocab:
                features[vocab[key].index, :] = np.array(value)
        
    with graph.as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)

        if feature_dic is not None:
            node_features = tf.Variable(features, name='node_features', trainable=False)
            feature_weights = tf.Variable(tf.truncated_normal([feature_dim, embedding_size], stddev=1.0))

            embed_trans = tf.Variable(tf.truncated_normal([feature_dim, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            u_embed_trans = tf.Variable(tf.truncated_normal([edge_type_count, feature_dim, embedding_u_size], stddev=1.0 / math.sqrt(embedding_size)))
        else:
            node_embeddings = tf.Variable(tf.random_uniform([num_nodes, embedding_size], -1.0, 1.0))
            node_type_embeddings = tf.Variable(tf.random_uniform([num_nodes, u_num, embedding_u_size], -1.0, 1.0))

        trans_weights = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, embedding_size // att_head], stddev=1.0 / math.sqrt(embedding_size)))
        trans_weights_s1 = tf.Variable(tf.truncated_normal([edge_type_count, embedding_u_size, dim_a], stddev=1.0 / math.sqrt(embedding_size)))
        trans_weights_s2 = tf.Variable(tf.truncated_normal([edge_type_count, dim_a, att_head], stddev=1.0 / math.sqrt(embedding_size)))
        nce_weights = tf.Variable(tf.truncated_normal([num_nodes, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([num_nodes]))

        # Input data
        train_inputs = tf.placeholder(tf.int32, shape=[None])
        train_labels = tf.placeholder(tf.int32, shape=[None, 1])
        train_types = tf.placeholder(tf.int32, shape=[None])
        node_neigh = tf.placeholder(tf.int32, shape=[None, edge_type_count, neighbor_samples])
        
        # Look up embeddings for nodes
        if feature_dic is not None:
            node_embed = tf.nn.embedding_lookup(node_features, train_inputs)
            node_embed = tf.matmul(node_embed, embed_trans)
        else:
            node_embed = tf.nn.embedding_lookup(node_embeddings, train_inputs)
        
        if feature_dic is not None:
            node_embed_neighbors = tf.nn.embedding_lookup(node_features, node_neigh)
            node_embed_tmp = tf.concat([tf.matmul(tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, 0], [-1, 1, -1, -1]), [-1, feature_dim]), tf.reshape(tf.slice(u_embed_trans, [i, 0, 0], [1, -1, -1]), [feature_dim, embedding_u_size])) for i in range(edge_type_count)], axis=0)
            node_type_embed = tf.transpose(tf.reduce_mean(tf.reshape(node_embed_tmp, [edge_type_count, -1, neighbor_samples, embedding_u_size]), axis=2), perm=[1,0,2])
        else:
            node_embed_neighbors = tf.nn.embedding_lookup(node_type_embeddings, node_neigh)
            node_embed_tmp = tf.concat([tf.reshape(tf.slice(node_embed_neighbors, [0, i, 0, i, 0], [-1, 1, -1, 1, -1]), [1, -1, neighbor_samples, embedding_u_size]) for i in range(edge_type_count)], axis=0)
            node_type_embed = tf.transpose(tf.reduce_mean(node_embed_tmp, axis=2), perm=[1,0,2])

        trans_w = tf.nn.embedding_lookup(trans_weights, train_types)
        trans_w_s1 = tf.nn.embedding_lookup(trans_weights_s1, train_types)
        trans_w_s2 = tf.nn.embedding_lookup(trans_weights_s2, train_types)
        
        attention = tf.reshape(tf.nn.softmax(tf.reshape(tf.matmul(tf.tanh(tf.matmul(node_type_embed, trans_w_s1)), trans_w_s2), [-1, u_num])), [-1, att_head, u_num])
        node_type_embed = tf.matmul(attention, node_type_embed)
        node_embed = node_embed + tf.reshape(tf.matmul(node_type_embed, trans_w), [-1, embedding_size])

        if feature_dic is not None:
            node_feat = tf.nn.embedding_lookup(node_features, train_inputs)
            node_embed = node_embed + tf.matmul(node_feat, feature_weights)

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

        # Optimizer.
        optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)

        # Add ops to save and restore all the variables.
        # saver = tf.train.Saver(max_to_keep=20)

        merged = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

        # Initializing the variables
        init = tf.global_variables_initializer()

    # Launch the graph
    print("Optimizing")
    
    with tf.Session(graph=graph) as sess:
        writer = tf.summary.FileWriter("./runs/" + log_name, sess.graph) # tensorboard --logdir=./runs
        sess.run(init)

        print('Training')
        g_iter = 0
        best_score = 0
        test_score = (0.0, 0.0, 0.0)
        patience = 0
        for epoch in range(epochs):
            random.shuffle(train_pairs)
            batches = get_batches(train_pairs, neighbors, batch_size)

            data_iter = tqdm(batches,
                            desc="epoch %d" % (epoch),
                            total=(len(train_pairs) + (batch_size - 1)) // batch_size,
                            bar_format="{l_bar}{r_bar}")
            avg_loss = 0.0

            for i, data in enumerate(data_iter):
                feed_dict = {train_inputs: data[0], train_labels: data[1], train_types: data[2], node_neigh: data[3]}
                _, loss_value, summary_str = sess.run([optimizer, loss, merged], feed_dict)
                writer.add_summary(summary_str, g_iter)

                g_iter += 1

                avg_loss += loss_value

                if i % 5000 == 0:
                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "avg_loss": avg_loss / (i + 1),
                        "loss": loss_value
                    }
                    data_iter.write(str(post_fix))
            
            final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))
            for i in range(edge_type_count):
                for j in range(num_nodes):
                    final_model[edge_types[i]][index2word[j]] = np.array(sess.run(last_node_embed, {train_inputs: [j], train_types: [i], node_neigh: [neighbors[j]]})[0])
            valid_aucs, valid_f1s, valid_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            for i in range(edge_type_count):
                if args.eval_type == 'all' or edge_types[i] in args.eval_type.split(','):
                    tmp_auc, tmp_f1, tmp_pr = evaluate(final_model[edge_types[i]], valid_true_data_by_edge[edge_types[i]], valid_false_data_by_edge[edge_types[i]])
                    valid_aucs.append(tmp_auc)
                    valid_f1s.append(tmp_f1)
                    valid_prs.append(tmp_pr)

                    tmp_auc, tmp_f1, tmp_pr = evaluate(final_model[edge_types[i]], testing_true_data_by_edge[edge_types[i]], testing_false_data_by_edge[edge_types[i]])
                    test_aucs.append(tmp_auc)
                    test_f1s.append(tmp_f1)
                    test_prs.append(tmp_pr)
            print('valid auc:', np.mean(valid_aucs))
            print('valid pr:', np.mean(valid_prs))
            print('valid f1:', np.mean(valid_f1s))

            average_auc = np.mean(test_aucs)
            average_f1 = np.mean(test_f1s)
            average_pr = np.mean(test_prs)

            cur_score = np.mean(valid_aucs)
            if cur_score > best_score:
                best_score = cur_score
                test_score = (average_auc, average_f1, average_pr)
                patience = 0
            else:
                patience += 1
                if patience > args.patience:
                    print('Early Stopping')
                    break
    return test_score

   
if __name__ == "__main__":
    args = parse_args()
    file_name = args.input
    print(args)
    if args.features is not None:
        feature_dic = load_feature_data(args.features)
    else:
        feature_dic = None

    log_name = file_name.split('/')[-1] + f'_evaltype_{args.eval_type}_b_{args.batch_size}_e_{args.epoch}'

    training_data_by_type = load_training_data(file_name + '/train.txt')
    valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(file_name + '/valid.txt')
    testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(file_name + '/test.txt')

    average_auc, average_f1, average_pr = train_model(training_data_by_type, feature_dic, log_name + '_' + time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time())))

    print('Overall ROC-AUC:', average_auc)
    print('Overall PR-AUC', average_pr)
    print('Overall F1:', average_f1)
