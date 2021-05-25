#!/usr/bin/python
from __future__ import division
from __future__ import print_function

import time
import os
import sys


import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.manifold import MDS
import networkx as nx

from optimizer import Optimizer
from input_data import load_data, load_masked_test_edges, load_masked_test_edges_for_kfold, load_data_semisup
from model import DLSM
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# Settings
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.set_random_seed(1234)
np.random.seed(1234)
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

#from utils import test_kl
#test_kl()

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Max number of epochs to train. Training may stop early if validation-error is not decreasing')
flags.DEFINE_string('encoder', '64_32_64', 'Number of units in encoder layers')
flags.DEFINE_string('decoder', '25_50', 'Number of units in decoder layers')
flags.DEFINE_integer('latent_dim', 50, 'Dimension of latent space (readout layer)')

flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('model', 'DLSM', 'Model to use')
flags.DEFINE_string('dataset', 'kohonen', 'Dataset string: cora, citeseer, pubmed, 20ng, llawyers_friends, llawyers_co-work, llawyers_adv, yeast, nips12, nips234, protein230')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('directed', 1, 'Whether the network is directed (1) or not (0).')
#flags.DEFINE_float('alpha_prior', 10., 'Prior alpha for Gamma')
flags.DEFINE_float('beta', 1., 'Posterior beta for Gamma')
flags.DEFINE_float('v0', 0.9, 'Prior parameter for steak-breaking IBP')
flags.DEFINE_float('alpha0', 10., 'Prior Alpha for Beta')
flags.DEFINE_float('p_prior', 0.2, 'Prior probability for Bernoulli')
flags.DEFINE_float('temp_prior', 0.5, 'Prior temperature for concrete distribution')
flags.DEFINE_float('temp_post', 1., 'Posterior temperature for concrete distribution')
flags.DEFINE_float('gating_weight', 20, 'Dimension of the gating weights for GNN')

# Not using K-fold
flags.DEFINE_integer('use_k_fold', 0, 'Whether to use k-fold cross validation')
flags.DEFINE_integer('k', 5, 'how many folds for cross validation.')
#flags.DEFINE_integer('save_pred_every', 5, 'Save summary after epochs')

flags.DEFINE_integer('early_stopping', 2000, 'how many epochs to train after last best validation')

# Split to use for evaluation
flags.DEFINE_integer('split_idx', 0, 'Dataset split (Total:10) 0-9')
# For experiment with missing data
flags.DEFINE_integer('missing_data', 0, 'Missing data experiment?')
flags.DEFINE_integer('missing_split_idx', 0, 'Dataset split for missing data experiments (split idx: 0 - 14).')

flags.DEFINE_integer('weighted_ce', 1, 'Weighted Cross Entropy: For class balance')
flags.DEFINE_integer('test', 0, 'Load model and run on test')

#options
flags.DEFINE_integer('use_kl_warmup', 0, 'Use a linearly increasing KL [0-1] coefficient -- see wu_beta in optimization.py')
flags.DEFINE_integer('use_x_warmup', 0, 'Use a linearly increasing [0-1] coefficient for multiplying with x_loss, annealing sort of -- see wu_x in optimization.py')

flags.DEFINE_float('bias_weight_1',1.0,'Multiplier for cross entropy loss for 1 label. See optimizer.py')
flags.DEFINE_string('expr_info','','Info about the experiment')
flags.DEFINE_float('lambda_mat_scale',0.1, 'Scale for Normal being used in initialization of scale parameter of lambda matrix')
flags.DEFINE_integer('cosine_norm',1,'Whether to use Cosine Normalized product instead of dot product for bilinear product')
flags.DEFINE_string('gpu_to_use','0','Which GPU to use. Leave blank to use None')
flags.DEFINE_integer('reconstruct_x',0,'Whether to separately reconstruct x')
flags.DEFINE_integer('link_prediction',1,'Whether to add link prediction loss')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss.') #5e-4

flags.DEFINE_integer('log_results',0,'Whether to log results')
flags.DEFINE_float('test_split',10.0,'Percentage of total edges to be kept as test data')
flags.DEFINE_float('val_split',5.0,'Percentage of total edges to be kept as val data')

#Use random_split for 20ng
flags.DEFINE_integer('random_split',0,'Whether to use random splits instead of fixed splits')

flags.DEFINE_integer('semisup_train',0,'Whether to perform semisupervised classification training as well')
flags.DEFINE_integer('mc_samples', 1, 'No. of MC samples for calculating gradients')

flags.DEFINE_string('data_type','binary','Type of data: binary, count')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_to_use

save_path_disk = "data/DLSM/data_models"

graph_dir = save_path_disk + '/DLSM/' + dataset_str + '/'
save_dir =  save_path_disk + '/DLSM/' + dataset_str +'/split_'+ str(FLAGS.split_idx) + '/' + model_str + "/" + FLAGS.encoder + "/"
if not os.path.exists(save_dir):
        os.makedirs(save_dir)

k_fold_str = '_no-k-fold'
if FLAGS.use_k_fold:
    k_fold_str = str(FLAGS.k)+'-fold'

#Let's start time here
start_time = time.time()

#if dataset_str == "20ng":
#    assert FLAGS.random_split == 1

# Load data. Raw adj is NxN Matrix and Features is NxF Matrix. Using sparse matrices here (See scipy docs). 
adj, features, feature_presence, directed = load_data(dataset_str)

# Set diagonal entries to be 0
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape = adj.shape)

if(FLAGS.semisup_train):
    y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_semisup(dataset_str)
    num_classes = y_train.shape[1]


#if(feature_presence == 0):
#   #save user from inadvertant errors
#   FLAGS.reconstruct_x = 0 

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
#adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape = adj_orig.shape)
adj_orig.eliminate_zeros()

print ("Adj Original Matrix: " + str(adj_orig.shape))
print ("Features Shape: " + str(features.shape))

num_nodes = adj_orig.shape[0]
features_shape = features.shape[0]
if FLAGS.features == 0:
        features = sp.identity(features_shape)  # featureless

pos_weight_feats = float(features.shape[0] * features.shape[1] - features.sum()) / features.sum()
norm_feats = features.shape[0] * features.shape[1] / float((features.shape[0] * features.shape[1] - features.sum()) * 2) # (N+P) x (N+P) / (N)

# feature sparse matrix to tuples 
features = sparse_to_tuple(features.tocoo())

def get_label_pred(sess, placeholders, feed_dict, model, S=2):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['is_training']: False})
    if model_str == 'DLSM':
        #get S posterior samples -> get S reconstructions
        op = np.zeros([num_nodes, num_classes])
        for i in range(S): 
            outs = sess.run([model.z], feed_dict=feed_dict)
            op += outs[0]
            #adj_rec = adj_rec + outs[3]
    return op/S

def get_score_matrix(sess, placeholders, feed_dict, model, S=5):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['is_training']: False})
        
    adj_rec = np.zeros([num_nodes, num_nodes])

    if model_str == 'DLSM':
        #get S posterior samples -> get S reconstructions
        for i in range(S): 
            outs = sess.run([model.reconstructions], feed_dict = feed_dict)
            #print (outs[0])
            #print (outs[1])
            #outs_list.append(outs)
            #adj_rec, z_activated = monte_carlo_sample(outs[0], outs[1], outs[2], FLAGS.temp_post, S, sigmoid)
            #adj_rec = adj_rec + outs[3]

            adj_rec += np.reshape(outs[0], (num_nodes, num_nodes))

    #average
    adj_rec = adj_rec/S
    
    return adj_rec

"""
Get Semi-Supervised training accuracy on test set
"""
def get_semisup_acc(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = np.equal(np.argmax(preds, 1), np.argmax(labels, 1))
    accuracy_all = correct_prediction.astype(float)
    mask = mask.astype(float)
    mask /= np.mean(mask)
    accuracy_all *= mask
 
    return np.mean(accuracy_all)

"""
Get ROC score and average precision
"""
def get_roc_score(adj_rec, edges_pos, edges_neg, emb=None):
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    preds = []
    pos = []
    for e in edges_pos:
        #preds.append(sigmoid(adj_rec[e[0], e[1]]))
        preds.append(adj_rec[e[0], e[1]])
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        #preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        preds_neg.append(adj_rec[e[0], e[1]])
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    # Compute precision recall curve 
    precision, recall, _ = precision_recall_curve(labels_all, preds_all)
    
    auc_pr = auc(recall, precision)
    #auc_prm = auc_pr_m(preds_all, labels_all)
    #print (str(auc_pr))
    #print (str(auc_prm))
    #sys.exit()
    
    return roc_score, ap_score, auc_pr

def auc_pr_m(probs, true_labels):

        #prob_1 = probs*true_labels + (1 - probs)*(1 - true_labels)
        prob_1 = probs
        
        isort = np.argsort(-1*prob_1) # descend

        #[dummy, isort] = np.sort(prob_1, 'descend')
        precision = np.cumsum( true_labels[isort] ) / np.arange(1, len(prob_1)+1)
        recall    = np.cumsum( true_labels[isort] ) / np.sum( true_labels )

        print (type(recall))
        print (recall.shape)

        print (recall)
        print (precision)
        
        recall = np.insert(recall, 0, 0)
        precision = np.insert(precision, 0, 1)
        
        #area = trapz([0,recall],[1,precision]) %in matlab
        area = np.trapz(precision,recall)

        return area

# create_model 
def create_model(placeholders, adj, features):

    num_nodes = adj.shape[0]
    num_features = features[2][1]
    features_nonzero = features[1].shape[0] # Can be used for dropouts. See GraphConvolutionSparse
    
    #print(num_features)

    # Create model
    model = None
    if model_str == 'DLSM':
        if FLAGS.semisup_train:
            model = DLSM(placeholders, num_features, num_nodes, features_nonzero, num_classes, mc_samples = FLAGS.mc_samples)
        else:
            model = DLSM(placeholders, num_features, num_nodes, features_nonzero, mc_samples = FLAGS.mc_samples)

    """
    if num_nodes > 10000:
        edges_for_loss = None
    else:
        edges_for_loss = placeholders['edges_for_loss']
    """
    edges_for_loss = placeholders['edges_for_loss']
    
    # Optimizer
    with tf.compat.v1.name_scope('optimizer'):
        if model_str == 'DLSM':
            if FLAGS.semisup_train:
                opt = Optimizer(labels=tf.reshape(tf.sparse.to_dense(placeholders['adj_orig'],
                                                                            validate_indices=False), [-1]),
                                model=model, num_nodes=num_nodes,
                                pos_weight=placeholders['pos_weight'],
                                norm=placeholders['norm'],
				weighted_ce = FLAGS.weighted_ce,
                                edges_for_loss=edges_for_loss,
                                epoch=placeholders['epoch'],
                                features = tf.reshape(tf.sparse.to_dense(placeholders['features'], validate_indices = False), [-1]),
                                norm_feats = placeholders['norm_feats'],
                                pos_weight_feats = placeholders['pos_weight_feats'],
                                node_labels = placeholders['node_labels'],
                                node_labels_mask = placeholders['node_labels_mask'],
                                start_semisup = placeholders['start_semisup'])
            else:
                opt = Optimizer(labels=tf.reshape(tf.sparse.to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
                                model=model,
                                #z_init = placeholders['z_init'],
                                num_nodes=num_nodes,
                                pos_weight=placeholders['pos_weight'],
                                norm=placeholders['norm'],
				                weighted_ce = FLAGS.weighted_ce,
                                edges_for_loss=edges_for_loss,
                                epoch=placeholders['epoch'],
                                features = tf.reshape(tf.sparse.to_dense(placeholders['features'], validate_indices = False), [-1]),
                                norm_feats = placeholders['norm_feats'],
                                pos_weight_feats = placeholders['pos_weight_feats'])


    return model, opt

def train(placeholders, model, opt, adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, features, sess, name="single_fold"):

    adj = adj_train
    
    # This will be calculated for every fold
    # pos_weight and norm should be tensors
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum() # N/P
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2) # (N+P) x (N+P) / (N)


    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Some preprocessing. adj_norm is D^(-1/2) x adj x D^(-1/2)
    adj_normalized = preprocess_graph(adj)
    """
    G = nx.from_numpy_matrix(adj.toarray()) 
    paths = nx.shortest_path(G)
    geo_distances = np.zeros(adj.shape)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            if j in paths[i].keys():
                geo_distances[i, j] = len(paths[i][j])
    #geo_distances[geo_distances == 0] = np.max(geo_distances) + 1
    geo_distances[geo_distances == 0] = np.max(geo_distances)
    geo_distances -= 1

    #z_init = MDS(n_components = FLAGS.latent_dim, dissimilarity = 'precomputed').fit_transform(geo_distances)
    z_init = mds(geo_distances, n_dims = FLAGS.latent_dim)
    """

    # get summaries
    #summaries = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)
    #for var in tf.compat.v1.trainable_variables():
    #        summaries.append(tf.compat.v1.summary.histogram(var.op.name, var))
    #summary_op = tf.compat.v1.summary.merge(summaries)

    # initialize summary_writer
    #summary_writer = tf.compat.v1.summary.FileWriter(graph_dir, sess.graph)
    #meta_graph_def = tf.compat.v1.train.export_meta_graph(filename=graph_dir+'/model.meta')
    #print("GRAPH IS  SAVED")
    #sys.stdout.flush()
    
    # session initialize
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()

    val_roc_score = []
    best_validation = 0.0
    """
    num_nodes = adj.shape[0]
    if num_nodes < 10000:
        edges_for_loss = np.arange(num_nodes * num_nodes)
        ignore_edges = []
        edges_to_ignore = np.concatenate((val_edges, val_edges_false, test_edges, test_edges_false), axis=0)
        for e in edges_to_ignore:
                ignore_edges.append(e[0]*num_nodes+e[1])
        edges_for_loss = np.delete(edges_for_loss, ignore_edges, 0)
    else:
        edges_for_loss = []
    """

    edges_for_loss = np.ones((num_nodes * num_nodes), dtype = np.float32)
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
        #feed_dict = construct_feed_dict(adj_normalized, adj_label, features, placeholders)
        feed_dict = dict()
        # adj_normalized, adj_label and features are tuple with coords, value, and shape. From coo matrix construct feed dictionary
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_normalized})
        feed_dict.update({placeholders['adj_orig']: adj_label})
        feed_dict.update({placeholders['edges']: train_edges})
        #feed_dict.update({placeholders['z_init']: z_init})
        #feed_dict.update({placeholders['indices_sender']: ind_sender})
        #feed_dict.update({placeholders['indices_receiver']: ind_receiver})
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['is_training']: True})
        feed_dict.update({placeholders['norm']: norm})
        feed_dict.update({placeholders['pos_weight']: pos_weight})
        feed_dict.update({placeholders['edges_for_loss']: edges_for_loss})
        feed_dict.update({placeholders['epoch']: epoch})
        feed_dict.update({placeholders['norm_feats']: norm_feats})
        feed_dict.update({placeholders['pos_weight_feats']: pos_weight_feats})

        if(FLAGS.semisup_train):
            feed_dict.update({placeholders['node_labels']: y_train})
            feed_dict.update({placeholders['node_labels_mask']: train_mask})
            
            #start semisup after sometime ?
            if epoch > 0.6 * FLAGS.epochs:
                feed_dict.update({placeholders['start_semisup']: 1.})
            else:
                feed_dict.update({placeholders['start_semisup']: 1.})

        # Run single weight update


        """
        if epoch % FLAGS.save_pred_every == 0:
                outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.kl_term, model.reconstructions, model.clipped_logit, model.posterior_theta_params, model.shape_d, model.theta, opt.regularization, model.prior_theta_params, model.lambda_mat, summary_op], feed_dict=feed_dict)
                summary = outs[-1]
                summary_writer.add_summary(summary, epoch)
        else:

        """
        #outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.kl_term, model.reconstructions, model.clipped_logit, model.posterior_theta_params, model.shape_d, model.theta, opt.regularization, model.prior_theta_params, model.lambda_mat, opt.grads_vars, opt.nll, opt.kl_term, opt.x_loss, opt.semisup_loss, opt.semisup_acc, model.z, opt.check, model.phi, opt.grads_vars], feed_dict=feed_dict)
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.kl, model.reconstructions, model.posterior_theta_param, model.theta, opt.regularization, opt.grads_vars, opt.nll, opt.x_loss, opt.semisup_loss, opt.semisup_acc, model.theta_decoder], feed_dict = feed_dict)
        #outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.kl, model.reconstructions, model.posterior_theta_param, model.theta, opt.regularization, opt.grads_vars, opt.nll, opt.x_loss, opt.semisup_loss, opt.semisup_acc], feed_dict = feed_dict)
        
        #outs = sess.run([model.reconstructions, model.clipped_logit, model.posterior_theta_params, model.shape_d, model.theta, model.prior_theta_params, model.lambda_mat, model.x_recon], feed_dict=feed_dict)
        
        #print (outs[-1])
        #print (outs[-2])
        #print (outs[-3])
        #print (outs[-1].shape)
        # Compute average loss
        
        avg_cost = outs[1]
        avg_accuracy = outs[2]
        kl = outs[3]
        #lambda_mat = outs[11]
        reconstructions = outs[4]
        #clipped_logit = outs[5]
        posterior_theta_param = outs[5]
        #shape_d = outs[7]
        theta = outs[6]
        regularization = outs[7]
        #prior_theta_params = outs[10];
        nll = outs[9]
        x_loss = outs[10]
        semisup_loss = outs[11]
        semisup_acc = outs[12]
        theta_decoder = outs[13]
        #model_z = outs[16]
        
        #phi = outs[16]
        #g_v = outs[21]

        #print (phi)
        #print(g_v)
        print (np.min(reconstructions), np.max(reconstructions), np.sum(reconstructions))
        if(np.isnan(kl)):
            print (posterior_theta_param)
            sys.exit()

        #print (avg_cost)
        print ('KL: ', kl, 'X_loss: ', x_loss, 'semisup_loss: ', semisup_loss, 'semisup train acc: ',semisup_acc, 'NLL: ', nll)
        #print (reconstructions)
        #print(outs[13])
        #print(outs[14])
        
               
        if True:#avg_accuracy > 0.9 or model_str == 'gcn_vae':

                #Validation
                adj_rec = get_score_matrix(sess, placeholders, feed_dict, model, S=2)
                roc_curr, ap_curr, _  = get_roc_score(adj_rec, val_edges, val_edges_false)
        
                print("Epoch:", '%03d' % (epoch + 1), "cost=", "{:.3f}".format(avg_cost), "kl=", "{:.3f}".format(outs[3]), "reg=", "{:.4f}".format(regularization), 
                      "train_acc=", "{:.3f}".format(avg_accuracy), "val_roc=", "{:.3f}".format(roc_curr), "val_ap=", "{:.3f}".format(ap_curr), 
                      "time=", "{:.2f}".format(time.time() - t))

                roc_curr = round(roc_curr, 3)
                val_roc_score.append(roc_curr)

                if roc_curr > best_validation:
                        # save model
                        print ('Saving model')
                        saver.save(sess=sess, save_path=save_dir+name)
                        best_validation = roc_curr
                        last_best_epoch = 0

                #if last_best_epoch > FLAGS.early_stopping and best_validation - roc_curr <= 0.003:
                if last_best_epoch > FLAGS.early_stopping:
                        break
                else:
                        last_best_epoch += 1
        else:
                print("Training Epoch:", '%03d' % (epoch + 1), "cost=", "{:.3f}".format(avg_cost), #"reg=", "{:.1f}".format(regularization),
                      "train_acc=", "{:.3f}".format(avg_accuracy), "time=", "{:.2f}".format(time.time() - t))

    print("Optimization Finished!")
    val_max_index = np.argmax(val_roc_score)
    print('Validation ROC Max: {:.3f} at Epoch: {:04d}'.format(val_roc_score[val_max_index], val_max_index))

    adj_score = get_score_matrix(sess, placeholders, feed_dict, model)
   
    #repeat for the sake of analysis
    roc_score, ap_score, auc_pr = get_roc_score(adj_score, test_edges, test_edges_false)
    
    if(FLAGS.semisup_train):
        model_z = get_label_pred(sess, placeholders, feed_dict, model)

    # Use this code for qualitative analysis
     
    # Use this code for qualitative analysis
    if FLAGS.missing_data:
        qual_file = 'data/qual_missing_' + dataset_str + '_' + model_str + k_fold_str + FLAGS.encoder + '_' + FLAGS.decoder
    else:
        qual_file = 'data/qual_' + dataset_str + '_' + model_str + k_fold_str + FLAGS.encoder + '_' + FLAGS.decoder + '_' + str(FLAGS.latent_dim)
    
    #theta_save = [layer_params for layer_params in theta]
    theta_save = [[param.tolist() for param in layer] for layer in theta]
    #phi_save = [layer_params.tolist() for layer_params in phi]

    # layer(s) --> nodes --> param(s)
    #Need to save phis too?
    #posterior_theta_param_save = [[[node_params.tolist() for node_params in param] for param in layer] for layer in posterior_theta_param]
    posterior_theta_param_save = [[param.tolist() for param in layer] for layer in posterior_theta_param]
    #prior_theta_param_save = [[[node_params.tolist() for node_params in param] for param in layer] for layer in prior_theta_param]
    #np.savez(qual_file, theta = theta_save, posterior_theta_params = posterior_theta_params_save, lambda_mat = lambda_mat, roc_score = roc_score, ap_score = ap_score, auc_pr = auc_pr, expr_info = FLAGS.expr_info, adj_train = adj_train, phis = phi_save, reconstruction = reconstructions)       
    np.savez(qual_file, theta = theta_save, theta_decoder = theta_decoder, posterior_theta_param = posterior_theta_param_save, roc_score = roc_score, ap_score = ap_score, auc_pr = auc_pr, expr_info = FLAGS.expr_info, adj_train = adj_train, reconstruction = reconstructions)
    saver.restore(sess=sess, save_path=(save_dir+name))

    return adj_score

def load_model(placeholders, model, opt, adj_train, train_edges, test_edges, test_edges_false, features, sess, name="single_fold"):

        adj = adj_train
        # This will be calculated for every fold
        # pos_weight and norm should be tensors
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum() # N/P
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2) # (N+P) x (N+P) / (N)

        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        # Some preprocessing. adj_norm is D^(-1/2) x adj x D^(-1/2)
        adj_norm = preprocess_graph(adj)
    
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['edges']: train_edges})
        #feed_dict.update({placeholders['indices_sender']: ind_sender})
        #feed_dict.update({placeholders['indices_receiver']: ind_receiver})
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['is_training']: True})
        feed_dict.update({placeholders['norm']: norm})
        feed_dict.update({placeholders['pos_weight']: pos_weight})
        feed_dict.update({placeholders['norm_feats']: norm_feats})
        feed_dict.update({placeholders['pos_weight_feats']: pos_weight_feats})
        
        # Some preprocessing. adj_norm is D^(-1/2) x adj x D^(-1/2)
        adj_norm = preprocess_graph(adj)
        saver = tf.compat.v1.train.Saver()
        
        saver.restore(sess=sess, save_path=(save_dir+name))
        print ('Model restored')

        if (dataset_str == 'pubmed'): # decreasing samples. Num of nodes high
                S = 5
        else:
                S = 15
        
        adj_score = get_score_matrix(sess, placeholders, feed_dict, model, S=S)

        return adj_score

def main():

    num_nodes = adj_orig.shape[0]
    print ("Model is " + model_str)

    # Define placeholders
    if FLAGS.semisup_train:
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
            'node_labels':  tf.compat.v1.placeholder(tf.float32),
            'node_labels_mask':  tf.compat.v1.placeholder(tf.int32),
            'start_semisup':tf.compat.v1.placeholder(tf.float32)
        }
    else:
        placeholders = {
            'features': tf.compat.v1.sparse_placeholder(tf.float32),
            'adj': tf.compat.v1.sparse_placeholder(tf.float32),
            'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
            'edges': tf.compat.v1.placeholder(tf.int64, shape=(None, 2)),
            #'z_init': tf.compat.v1.placeholder(tf.float32, shape=(None, FLAGS.latent_dim)),
            #'indices_sender': tf.compat.v1.placeholder(tf.int64, shape = (None, 1)),
            #'indices_receiver': tf.compat.v1.placeholder(tf.int64, shape = (None, 1)),
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
    
    if FLAGS.use_k_fold: # Don't use k-fold for large dataset

        k_adj_train, k_train_edges, k_val_edges, k_val_edges_false, test_edges, test_edges_false = load_masked_test_edges_for_kfold(dataset_str, FLAGS.k, FLAGS.split_idx)
        #k_adj_train, k_train_edges, k_val_edges, k_val_edges_false, test_edges, test_edges_false = mask_test_edges_for_kfold(adj, FLAGS.k, all_edge_idx)

        all_adj_scores = np.zeros((num_nodes, num_nodes))
        for k_idx in range(FLAGS.k):
            print (str(k_idx) + " fold")

            adj_train = k_adj_train[k_idx]
            train_edges = k_train_edges[k_idx]
            val_edges = k_val_edges[k_idx]
            val_edges_false = k_val_edges_false[k_idx]

            if FLAGS.test:
                    adj_score  = load_model(placeholders, model, opt, adj_train, train_edges, test_edges, test_edges_false,
                                                        features, sess, name="k-fold-%d"%(k_idx+1))
            else:
                    adj_score, model_z = train(placeholders, model, opt, adj_train, train_edges, val_edges, val_edges_false,
                                                   test_edges, test_edges_false, features, sess, name="k-fold-%d"%(k_idx+1))
            
            all_adj_scores += adj_score

        all_adj_scores /= FLAGS.k
        roc_score, ap_score, auc_pr = get_roc_score(all_adj_scores, test_edges, test_edges_false)

    else:
        
        if FLAGS.random_split:
            adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, FLAGS.test_split, FLAGS.val_split, None)
        else:
            if FLAGS.missing_data:
                adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = load_masked_test_edges(dataset_str+"/missing", FLAGS.missing_split_idx)
            else:
                adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = load_masked_test_edges(dataset_str, FLAGS.split_idx)
            
            adj_train = adj_train[0]
            #adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, None)
        
        if FLAGS.test:
            adj_score  = load_model(placeholders, model, opt, adj_train, train_edges, test_edges,
                                                    test_edges_false, features, sess)
        else:
            adj_score = train(placeholders, model, opt, adj_train, train_edges, val_edges,
                                               val_edges_false, test_edges, test_edges_false, features, sess)

        roc_score, ap_score, auc_pr = get_roc_score(adj_score, test_edges, test_edges_false)
        all_adj_scores = adj_score
        
        if(FLAGS.semisup_train):
            semisup_acc = get_semisup_acc(model_z, y_test, test_mask)

    
    # Testing
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
    print('Test AUC PR Curve: ' + str(auc_pr))

    if(FLAGS.semisup_train):
        print('Test Acc. :', str(semisup_acc))

    if FLAGS.log_results:
        results_log_file = save_path_disk + "results_log_"  + dataset_str + '_' + model_str + k_fold_str + FLAGS.encoder + '.log'

        #if path exists
        if not os.path.exists(os.path.dirname(results_log_file)):
            os.makedirs(os.path.dirname(results_log_file))
        
        end_time = time.time()
        time_taken = end_time - start_time

        with open(results_log_file,'a') as rlog_file:
            rlog_file.write('Split: {}\nROC: {}\nAP:{}\nAUC-PR: {}\nTime-taken (s): {}\n\n'.format(FLAGS.split_idx, str(roc_score),str(ap_score),str(auc_pr), str(time_taken)))

            #if FLAGS.semisup_train:
            #    rlog_file.write('Semisup-training accuracy: {}\n'.format(str(semisup_acc)))
        #print ("*"*10)
        print ("\n")
    
    """
    Qualitative analysis. Overlapping communities

    k_fold_str = '_no-k-fold'
    if FLAGS.use_k_fold:
            k_fold_str = ''
            
    qual_file = '../data/qual_adj_score_' + dataset_str + '_' + model_str + k_fold_str
    npsavez(qual_file, adj_score=all_adj_scores)
    """

if __name__ == '__main__':
    main()

