#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import time
import os
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from scipy.stats import mode
import collections
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.cluster import KMeans

from optimizer import Optimizer
from input_data import load_data, load_training_data, load_node_labels
from model import DLSM, DLSM_D
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple
from utils import compute_graph_statistics

# Settings
tf.compat.v1.disable_v2_behavior()
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
# Experiment settings
flags.DEFINE_string('model', 'dlsm', 'Model to use: dlsm_d')
flags.DEFINE_string('dataset', 'political', 'Dataset string: email, wiki, google')
flags.DEFINE_integer('split_idx', 0, 'Dataset split (Total:10) 0-9')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('directed', 1, 'Whether the network is directed (1) or not (0).')
flags.DEFINE_integer('epochs', 2000, 'Max number of epochs to train. Training may stop early if validation-error is not decreasing')
flags.DEFINE_integer('early_stopping', 500, 'Number epochs to train after last best validation')
flags.DEFINE_integer('link_prediction', 1, 'Conduct link prediction')
flags.DEFINE_integer('community_detection', 0, 'Conduct community detection')
flags.DEFINE_integer('graph_generation', 0, 'Conduct graph generation')
flags.DEFINE_integer('comm_least_size', 0, 'Least number of nodes for each community (30 for email)')
flags.DEFINE_string('gpu_to_use', '0', 'Which GPU to use. Leave blank to use None')
# Model settings
flags.DEFINE_string('encoder', '32_64_128', 'Number of units in encoder layers')
flags.DEFINE_string('decoder', '50_100', 'Number of units in decoder layers')
flags.DEFINE_integer('latent_dim', 50, 'Dimension of latent space (readout layer)')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('use_kl_warmup', 0, 'Use a linearly increasing KL [0-1] coefficient -- see wu_beta in optimization.py')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('beta', 1., 'Posterior beta for Gamma')
flags.DEFINE_float('v0', 0.9, 'Prior parameter for steak-breaking IBP')
flags.DEFINE_float('temp_prior', 0.5, 'Prior temperature for concrete distribution')
flags.DEFINE_float('temp_post', 1., 'Posterior temperature for concrete distribution')
flags.DEFINE_integer('mc_samples', 1, 'No. of MC samples for calculating gradients')

model_str = FLAGS.model
dataset_str = FLAGS.dataset
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_to_use

save_dir =  "data/" + model_str + "/" + dataset_str +'/split_'+ str(FLAGS.split_idx) + '/' + FLAGS.encoder + "/"
if not os.path.exists(save_dir):
        os.makedirs(save_dir)

#Let's start time here
start_time = time.time()
# Load data. Raw adj is nxn Matrix and Features is nxp Matrix. Using sparse matrices here (See scipy docs). 
adj, features, feature_presence, directed = load_data(dataset_str)
# Set diagonal entries to be 0
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape = adj.shape)
# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig.eliminate_zeros()
print ("Adj Original Matrix: " + str(adj_orig.shape))
print ("Features Shape: " + str(features.shape))

num_nodes = adj_orig.shape[0]
features_shape = features.shape[0]
if FLAGS.features == 0:
    features = sp.identity(features_shape)  # featureless
pos_weight_feats = float(features.shape[0] * features.shape[1] - features.sum()) / features.sum()
norm_feats = features.shape[0] * features.shape[1] / float((features.shape[0] * features.shape[1] - features.sum()) * 2)
# feature sparse matrix to tuples 
features = sparse_to_tuple(features.tocoo())

def generate_graph(adj_score, edge_num):
    """
    Generate a binary graph from the input score matrix. 
    Ensures that there will be no singleton nodes.

    Parameters
    ----------
    adj_score: np.array of shape (N,N). The input adjacency scores.
    n_edges: int. The desired number of edges in the target graph.

    Returns
    -------
    The generated graph.
    """

    if len(adj_score.nonzero()[0]) < edge_num:
        adj_score[adj_score > 1] = 1
        return np.round(adj_score, 0)

    adj_gen = np.zeros(adj_score.shape) # initialize target graph    
    max_score_args_out = np.argmax(adj_score, axis = 1)
    max_score_args_in = np.argmax(adj_score, axis = 0)
    max_scores_out = np.max(adj_score, axis = 1)
    max_scores_in = np.max(adj_score, axis = 0)
    for i in range(adj_score.shape[0]):
        if max_scores_out[i] > max_scores_in[i]:
            adj_gen[i, max_score_args_out[i]] = 1
        else:
            adj_gen[max_score_args_in[i], i] = 1
    
    adj_gen = np.reshape(adj_gen, [-1])
    args_sorted = np.argsort(-np.reshape(adj_score, [-1]))
    i = 0
    while adj_gen.sum() <= edge_num:
        adj_gen[args_sorted[i]] = 1
        i += 1

    return np.reshape(adj_gen, adj_score.shape)

def get_score_matrix(sess, placeholders, feed_dict, model, S = 5):

    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['is_training']: False})
    
    adj_rec = np.zeros([num_nodes, num_nodes])
    # get S posterior samples -> get S reconstructions
    for i in range(S): 
        outs = sess.run([model.reconstructions], feed_dict = feed_dict)
        adj_rec += np.reshape(outs[0], (num_nodes, num_nodes))
    # average
    adj_rec = adj_rec / S
    
    return adj_rec

# Get AUC score and average precision for link prediction
def get_roc_score(adj_rec, edges_pos, edges_neg, emb=None):

    preds = []
    pos = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    
    return roc_score, ap_score

# Get statistics of directed graphs for graph generation
def get_statistics_score(adj_rec, adj_gt):

    statistics_pred = compute_graph_statistics(adj_rec)
    statistics_label = compute_graph_statistics(adj_gt)
    
    outdeg_score = abs(statistics_pred['d_out_max'] - statistics_label['d_out_max']) / statistics_label['d_out_max']
    indeg_score = abs(statistics_pred['d_in_max'] - statistics_label['d_in_max']) / statistics_label['d_in_max']
    tr_score = abs(statistics_pred['transitivity_rate'] - statistics_label['transitivity_rate']) / statistics_label['transitivity_rate']
    rr_score = abs(statistics_pred['reciprocity_rate'] - statistics_label['reciprocity_rate']) / statistics_label['reciprocity_rate']
    outpl_score = abs(statistics_pred['power_law_exp_out'] - statistics_label['power_law_exp_out']) / statistics_label['power_law_exp_out']
    inpl_score = abs(statistics_pred['power_law_exp_in'] - statistics_label['power_law_exp_in']) / statistics_label['power_law_exp_in']
    cc_score = abs(statistics_pred['clustering_coefficient'] - statistics_label['clustering_coefficient']) / statistics_label['clustering_coefficient']
    
    return outdeg_score, indeg_score, tr_score, rr_score, outpl_score, inpl_score, cc_score

# create_model 
def create_model(placeholders, adj, features):

    num_nodes = adj.shape[0]
    num_features = features[2][1]
    features_nonzero = features[1].shape[0] # Can be used for dropouts. See GraphConvolutionSparse

    # Create model
    model = None
    if model_str == 'dlsm':
        model = DLSM(placeholders, num_features, num_nodes, features_nonzero, mc_samples = FLAGS.mc_samples)
    elif model_str == 'dlsm_d':
        model = DLSM_D(placeholders, num_features, num_nodes, features_nonzero, mc_samples = FLAGS.mc_samples)
    else:
        raise NameError('No model named ' + model_str + '.')

    edges_for_loss = placeholders['edges_for_loss']
    
    # Optimizer
    with tf.compat.v1.name_scope('optimizer'):
        opt = Optimizer(labels=tf.reshape(tf.sparse.to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
                        model=model,
                        num_nodes=num_nodes,
                        pos_weight=placeholders['pos_weight'],
                        norm=placeholders['norm'],
                        edges_for_loss=edges_for_loss,
                        epoch=placeholders['epoch'],
                        model_str = model_str)

    return model, opt

def train(placeholders, model, opt, adj_train, adj_gt, val_edges, val_edges_false, test_edges, test_edges_false, features, sess):
    
    # pos_weight and norm should be tensors
    pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
    norm = adj_train.shape[0] * adj_train.shape[0] / float((adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) * 2)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Some preprocessing. adj_norm is D^(-1/2) x adj x D^(-1/2)
    adj_normalized = preprocess_graph(adj_train)
    
    # session initialize
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()

    val_roc_score = []
    best_validation = 0.0

    edges_for_loss = np.ones((num_nodes, num_nodes), dtype = np.float32)
    edges_for_loss[np.diag_indices_from(edges_for_loss)] = 0.0
    edges_to_ignore = np.concatenate((val_edges, val_edges_false, test_edges, test_edges_false), axis=0)
    for e in edges_to_ignore:
        edges_for_loss[e[0], e[1]] = 0.0
        if FLAGS.directed == 0:
            edges_for_loss[e[1], e[0]] = 0.0
    edges_for_loss = np.reshape(edges_for_loss, [-1])

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        # adj_normalized, adj_label and features are tuple with coords, value, and shape. From coo matrix construct feed dictionary
        feed_dict = construct_feed_dict(adj_normalized, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['is_training']: True})
        feed_dict.update({placeholders['norm']: norm})
        feed_dict.update({placeholders['pos_weight']: pos_weight})
        feed_dict.update({placeholders['edges_for_loss']: edges_for_loss})
        feed_dict.update({placeholders['epoch']: epoch})
        feed_dict.update({placeholders['norm_feats']: norm_feats})
        feed_dict.update({placeholders['pos_weight_feats']: pos_weight_feats})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.kl, model.reconstructions, model.posterior_theta_param, model.theta, model.theta_decoder], feed_dict = feed_dict)
        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]
        kl = outs[3]
        reconstructions = outs[4]
        posterior_theta_param = outs[5]
        theta = outs[6]
        theta_decoder = outs[7]
        #Validation
        adj_rec = get_score_matrix(sess, placeholders, feed_dict, model, S=2)
        roc_curr, ap_curr  = get_roc_score(adj_rec, val_edges, val_edges_false)
        
        print("Epoch:", '%03d' % (epoch + 1), "cost=", "{:.3f}".format(avg_cost), "kl=", "{:.3f}".format(kl), "train_acc=", "{:.3f}".format(avg_accuracy),
              "val_roc=", "{:.3f}".format(roc_curr), "val_ap=", "{:.3f}".format(ap_curr), "time=", "{:.2f}".format(time.time() - t))
              
        roc_curr = round(roc_curr, 3)
        val_roc_score.append(roc_curr)

        if roc_curr > best_validation:
            # save model
            print ('Saving model')
            saver.save(sess = sess, save_path = save_dir)
            best_validation = roc_curr
            last_best_epoch = 0

        if last_best_epoch > FLAGS.early_stopping and best_validation - roc_curr <= 0.003:
            break
        else:
            last_best_epoch += 1

    print("Optimization Finished!")
    val_max_index = np.argmax(val_roc_score)
    print('Validation ROC Max: {:.3f} at Epoch: {:04d}'.format(val_roc_score[val_max_index], val_max_index))

    # Testing
    adj_score = get_score_matrix(sess, placeholders, feed_dict, model)
    # link prediction
    if FLAGS.link_prediction:
        roc_score, ap_score = get_roc_score(adj_score, test_edges, test_edges_false)
        print('Test AUC score: ' + str(roc_score))
        print('Test AP score: ' + str(ap_score))
    # graph generation
    if FLAGS.graph_generation:
        adj_gen = generate_graph(adj_score, adj_train.sum())
        test_scores = get_statistics_score(adj_gen, adj_gt)
        print('Test max out-degree score: ' + str(test_scores[0]))
        print('Test max in-degree score: ' + str(test_scores[1]))
        print('Test transitivity rate score: ' + str(test_scores[2]))
        print('Test reciprocity rate score: ' + str(test_scores[3]))
        print('Test out power-law coefficient score: ' + str(test_scores[4]))
        print('Test in power-law coefficient score: ' + str(test_scores[5]))
        print('Test clustering coefficient score: ' + str(test_scores[6]))
     
    # Use this code for qualitative analysis
    qual_file = 'data/qual_' + dataset_str + str(FLAGS.split_idx) + '_' + model_str + '_' + FLAGS.encoder + '_' + FLAGS.decoder + '_' + str(FLAGS.latent_dim)
    # layer(s) --> nodes --> param(s)
    theta_save = [[param.tolist() for param in layer] for layer in theta]
    posterior_theta_param_save = [[param.tolist() for param in layer] for layer in posterior_theta_param]  
    np.savez(qual_file,
             theta = theta_save,
             theta_decoder = theta_decoder,
             posterior_theta_param = posterior_theta_param_save,
             reconstruction = reconstructions)
    saver.restore(sess = sess, save_path = save_dir)

    return theta_decoder[0]

def main():

    print ("Model is " + model_str)

    # Define placeholders
    placeholders = {
        'features': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'is_training': tf.compat.v1.placeholder(tf.bool),
        'norm': tf.compat.v1.placeholder(tf.float32),
        'pos_weight': tf.compat.v1.placeholder(tf.float32),
        'edges_for_loss': tf.compat.v1.placeholder(tf.float32),
        'epoch': tf.compat.v1.placeholder(tf.int32),
        'norm_feats': tf.compat.v1.placeholder(tf.float32),
        'pos_weight_feats': tf.compat.v1.placeholder(tf.float32),
    }

    model, opt = create_model(placeholders, adj, features)
    sess = tf.compat.v1.Session()
        
    adj_train, val_edges, val_edges_false, test_edges, test_edges_false = load_training_data(dataset_str, FLAGS.split_idx)
    adj_train = adj_train[0]
    
    emb = train(placeholders, model, opt, adj_train, adj_orig, val_edges, val_edges_false, test_edges, test_edges_false, features, sess)
    
    ## Community Detection ##
    if FLAGS.community_detection:
        
        labels = load_node_labels(dataset_str)
        label_count = collections.Counter(labels.tolist())
        # eliminate small communities
        comm_keep = np.array(list(label_count.keys()))[np.array(list(label_count.values())) > FLAGS.comm_least_size]
        z_keep = emb[np.in1d(labels, comm_keep), :]
        labels_keep = labels[np.in1d(labels, comm_keep)]
        num_classes = comm_keep.shape[0]
        print('Keep ' + str(num_classes) + ' communities.')

        km = KMeans(n_clusters = num_classes)
        print('Perform K-means for community detection with the learned embeddings..')
        clusters = km.fit_predict(z_keep) # cluster using K-means
        preds = np.zeros_like(clusters)
        # match the predicted labels and groundtruth labels
        for i in range(num_classes):
            # get a bool index matrix of the i-th clustering community
            mask = (clusters == i) 
            # find the most linkely community in the real labels
            preds[mask] = mode(labels_keep[mask])[0]
    
        f1_macro = f1_score(labels_keep, preds, average = 'macro')
        f1_micro = f1_score(labels_keep, preds, average = 'micro')
    
        print('Macro F1-score: ' + str(f1_macro))
        print('Micro F1-score: ' + str(f1_micro))

if __name__ == '__main__':
    main()
