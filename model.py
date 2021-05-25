from layers import FullConnection, GraphConvolution, GraphConvolutionSparse, PosInnerProductDecoder, LSMDecoder, DiagonalInnerProductDecoder, SparseLinearLayer
import tensorflow as tf
import numpy as np
from utils import *
from initializations import weight_variable_glorot

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
SMALL = 1e-16

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass
   
class DLSM(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, num_classes = 0, mc_samples=1, **kwargs):
        #super(DLSM, self).__init__(**kwargs)
        super().__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.outdegrees = tf.sparse.sparse_dense_matmul(self.adj, tf.ones([num_nodes, 1]))
        self.indegrees = tf.cond(pred = tf.equal(FLAGS.directed, 1), true_fn = lambda: tf.sparse.sparse_dense_matmul(tf.ones([1, num_nodes]), self.adj), false_fn = lambda: self.outdegrees)
        self.edges = placeholders['edges']
        #self.ind_sender = placeholders['indices_sender']
        #self.ind_receiver = placeholders['indices_receiver']
        self.dropout = placeholders['dropout']
        self.training = placeholders['is_training']
        #self.weighted_links = weighted_links

        self.encoder_layers = [int(x) for x in FLAGS.encoder.split('_')]
        self.decoder_layers = [int(x) for x in FLAGS.decoder.split('_')]
        self.num_encoder_layers = len(self.encoder_layers)
        self.num_decoder_layers = len(self.decoder_layers)
        
        self.prior_theta_param = []
        self.posterior_theta_param = []
        self.z = []
        self.num_classes = num_classes
        self.S = mc_samples #No. of MC samples
        self.a_val = np.log(np.exp(FLAGS.alpha0) - 1) # inverse softplus
        self.b_val = np.log(np.exp(1.) - 1)
        
        self.build()

    def get_regualizer_cost(self, regularizer):

        regularization = 0
        #regularization += self.last_layer.apply_regularizer(regularizer)
        
        for layer in self.layers:
            regularization += regularizer(layer.vars['weights'])# * FLAGS.weight_decay

        return regularization
    
    def _build(self):

        print('Build Dynamic Network....')

        self.posterior_theta_param = []
        self.h = []
        self.layers = []
        h = self.inputs

        # Upward Inference Pass
        for idx, encoder_layer in enumerate(self.encoder_layers):

            #act = lambda a: tf.nn.leaky_relu(a, alpha=0.2)
            
            #This selection is questionable. May not be much of effect in reality
            if FLAGS.semisup_train:
                act = tf.nn.relu
            else:
                #act = lambda x: x
                act = tf.nn.sigmoid
            #act = tf.nn.relu

            """
            if idx+1 == self.num_hidden_layers:
                act = lambda x:x
            else:
                act = lambda a: tf.nn.leaky_relu(a, alpha=0.2)
            """
            
            if idx == 0:
                
                gc = GraphConvolutionSparse(input_dim = self.input_dim,
                                            output_dim = encoder_layer,
                                            adj = self.adj,
                                            outdegrees = self.outdegrees,
                                            indegrees = self.indegrees,
                                            edges = self.edges,
                                            #ind_sender = self.ind_sender,
                                            #ind_receiver = self.ind_receiver,
                                            num_nodes = self.n_samples,
                                            features_nonzero = self.features_nonzero,
                                            act = lambda x: x,
                                            name = "conv_weight_input_" + str(idx),
                                            dropout = self.dropout,
                                            logging = self.logging)
                h = gc(self.inputs)
                #h = tf.compat.v1.check_numerics(h, 'h0 is nan')
                self.layers.append(gc)
                
                if(False and FLAGS.features==1 and FLAGS.reconstruct_x==1):
                    #feature transform
                    x_h = SparseLinearLayer(input_dim=self.input_dim,
                                        output_dim=encoder_layer,
                                        dropout=self.dropout,
                                        features_nonzero=self.features_nonzero,
                                        reuse_name = "conv_weight_x_"+ str(idx),
                                        reuse = True)(self.inputs)
                    x_h = tf.nn.relu(x_h)
                #self.h.append([h])
                if FLAGS.directed == 1:
                    self.h.append([h, h, h, h, h])
                else:
                    self.h.append([h, h, h, h])

            else:
                gc_mean = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                     output_dim = encoder_layer, #self.num_classes,
                                     adj = self.adj,
                                     outdegrees = self.outdegrees,
                                     indegrees = self.indegrees,
                                     edges = self.edges,
                                     #ind_sender = self.ind_sender,
                                     #ind_receiver = self.ind_receiver,
                                     num_nodes = self.n_samples,
                                     act = act,
                                     dropout = self.dropout,
                                     name = "conv_weight_mean_" + str(idx),
                                     logging = self.logging)
                h_mean = gc_mean(self.h[-1][0])
                #h_mean = tf.compat.v1.check_numerics(h_mean, 'h_mean is nan')
                self.layers.append(gc_mean)
                
                gc_std = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                     output_dim = encoder_layer, #self.num_classes,
                                     adj = self.adj,
                                     outdegrees = self.outdegrees,
                                     indegrees = self.indegrees,
                                     edges = self.edges,
                                     #ind_sender = self.ind_sender,
                                     #ind_receiver = self.ind_receiver,
                                     num_nodes = self.n_samples,
                                     act = act,
                                     dropout = self.dropout,
                                     name = "conv_weight_std_" + str(idx),
                                     logging = self.logging)
                h_std = gc_std(self.h[-1][1])
                #h_std = tf.compat.v1.check_numerics(h_std, 'h_std is nan')
                self.layers.append(gc_std)
                
                gc_pi = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                     output_dim = encoder_layer, #self.num_classes,
                                     adj = self.adj,
                                     outdegrees = self.outdegrees,
                                     indegrees = self.indegrees,
                                     edges = self.edges,
                                     #ind_sender = self.ind_sender,
                                     #ind_receiver = self.ind_receiver,
                                     num_nodes = self.n_samples,
                                     act = act,
                                     dropout = self.dropout,
                                     name = "conv_weight_pi_" + str(idx),
                                     logging = self.logging)
                #h = gc(self.h[-1])
                h_pi = gc_pi(self.h[-1][2])
                #h_pi = tf.compat.v1.check_numerics(h_pi, 'h_pi is nan')
                self.layers.append(gc_pi)
                
                gc_alpha_gam = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                     output_dim = encoder_layer, #self.num_classes,
                                     adj = self.adj,
                                     outdegrees = self.outdegrees,
                                     indegrees = self.indegrees,
                                     edges = self.edges,
                                     #ind_sender = self.ind_sender,
                                     #ind_receiver = self.ind_receiver,
                                     num_nodes = self.n_samples,
                                     act = act,
                                     dropout = self.dropout,
                                     name = "conv_weight_alpha_gam_" + str(idx),
                                     logging = self.logging)
                #h = gc(self.h[-1])
                #h_alpha = gc_alpha(self.h[-1][3])
                h_alpha_gam = gc_alpha_gam(self.h[-1][3])
                #h_alpha_gam = tf.compat.v1.check_numerics(h_alpha_gam, 'h_alpha is nan')
                self.layers.append(gc_alpha_gam)
                
                if FLAGS.directed == 1:
                    gc_alpha_del = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                     output_dim = encoder_layer, #self.num_classes,
                                     adj = self.adj,
                                     outdegrees = self.outdegrees,
                                     indegrees = self.indegrees,
                                     edges = self.edges,
                                     #ind_sender = self.ind_sender,
                                     #ind_receiver = self.ind_receiver,
                                     num_nodes = self.n_samples,
                                     act = act,
                                     dropout = self.dropout,
                                     name = "conv_weight_alpha_del_" + str(idx),
                                     logging = self.logging)
                    #h = gc(self.h[-1])
                    #h_alpha = gc_alpha(self.h[-1][3])
                    h_alpha_del = gc_alpha_del(self.h[-1][4])
                    #h_alpha = tf.compat.v1.check_numerics(h_alpha, 'h_alpha is nan')
                    self.layers.append(gc_alpha_del)
                    
                    self.h.append([h_mean, h_std, h_pi, h_alpha_gam, h_alpha_del])
                    
                else:
                    self.h.append([h_mean, h_std, h_pi, h_alpha_gam])
                
                if(False and FLAGS.features==1 and FLAGS.reconstruct_x==1):
                    #feature transform
                    x_h = FullConnection(input_dim=self.encoder_layers[idx-1],
                                        output_dim=encoder_layer,
                                        act = lambda x: tf.nn.relu(x),
                                        dropout=self.dropout,
                                        reuse_name = "conv_weight_"+str(idx),
                                        reuse = True)(x_h)

            #d = tf.nn.l2_normalize(d, axis=1)
            
            #h = tf.compat.v1.check_numerics(h, 'h is nan')

            # get Theta parameters
                
                #mean_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx], act = lambda x: tf.nn.leaky_relu(x, alpha = 0.2))
                mean_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: x)
                self.layers.append(mean_layer)
                z_mean = mean_layer(self.h[-1][0])
                #z_mean.append(mean_layer(self.h[-1][0][d]))
                #std_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx], act = lambda x: tf.nn.leaky_relu(x, alpha = 0.2))
                std_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: x)
                self.layers.append(std_layer)
                z_std = std_layer(self.h[-1][1])
                #z_std.append(std_layer(self.h[-1][0][d]))
                
                #pi_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx], act = lambda x: tf.nn.leaky_relu(x, alpha = 0.2))
                pi_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: x)
                self.layers.append(pi_layer)
                pi_logit = pi_layer(self.h[-1][2])
                #pi_logit = pi_layer(self.h[-1][1])
                
                alpha_gam_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: tf.nn.softplus(x))
                #alpha_gam_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx], act = lambda x: tf.nn.relu(x))
                self.layers.append(alpha_gam_layer)
                #alpha_gam = alpha_gam_layer(self.h[-1][3])
                alpha_gam = alpha_gam_layer(self.h[-1][3])
            
            #zeros = tf.zeros(self.decoder_layers[idx])
            #with tf.compat.v1.variable_scope('beta_para' + str(idx)):
                #a = tf.Variable(zeros, name = "alpha") + self.a_val
                #b = tf.Variable(zeros, name = "beta") + self.b_val
            #beta_a = tf.nn.softplus(a) 
            #beta_b = tf.nn.softplus(b) 
            #beta_a = tf.expand_dims(beta_a, 0) # dim: 1 × K^l
            #beta_b = tf.expand_dims(beta_b, 0) # dim: 1 × K^l
            #beta_a = tf.tile(beta_a, [self.n_samples, 1]) # dim: N × K^l
            #beta_b = tf.tile(beta_b, [self.n_samples, 1]) # dim: N × K^l
            
                if FLAGS.directed == 1:
                    alpha_del_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: tf.nn.softplus(x))
                #alpha_del_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx], act = lambda x: tf.nn.relu(x))
                    self.layers.append(alpha_del_layer)
                    alpha_del = alpha_del_layer(self.h[-1][4])
                    #alpha_del = alpha_del_layer(self.h[-1][2])
            
                    #self.posterior_theta_param.append([z_mean, z_std, pi_logit, alpha_gam, alpha_del]) 
                    self.posterior_theta_param.append([z_mean, z_std, pi_logit, alpha_gam, alpha_del])
                #self.posterior_theta_param.append([z_mean, z_std, pi_logit, alpha_gam, alpha_del, beta_a, beta_b]) 
                else:
                    #self.posterior_theta_param.append([z_mean, z_std, pi_logit, alpha_gam])
                    self.posterior_theta_param.append([z_mean, z_std, pi_logit, alpha_gam])
                #self.posterior_theta_param.append([z_mean, z_std, pi_logit, alpha_gam, beta_a, beta_b])

        # Downward Inference pass
        # Careful: The posterior order is reverse of prior (and theta samples).
        # We will invert array after downward pass is complete
        # Note that in formulation here, Weibull has shape,scale parametrization while gamma has shape,rate parametrization
        #prior_shape_const = tf.constant(10e-5, tf.float32)
        #prior_rate_const = tf.constant(10e-3, tf.float32)
        #prior_rate_const2 = tf.constant(10e-2, tf.float32)

        #Merged!
        #x_h = d

        self.theta_list = []      
        self.reconstructions_list = []
        self.posterior_theta_param_list = []
        self.prior_theta_param_list = []
        #self.phi = []
        self.reg_phi = 0.
        #posterior_theta_param = self.posterior_theta_param
        
        ###########################################################################
        #Take multiple MC samples
        for k in range(self.S):
            # Refresh
            
            # Downward Inference Pass
            #for idx in range(self.num_hidden_layers-2, -1, -1): # l = L-1, L-2, ..., 1
            self.theta = []
            self.prior_theta_param = []
            
            # Downward Inference Pass
            for idx, decoder_layer in enumerate(self.decoder_layers): # l = L-1, L-2, ..., 1
                
                if idx == 0:

                    #mean_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: tf.nn.leaky_relu(x, alpha = 0.2))
                    mean_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: x)
                    self.layers.append(mean_layer)
                    self.posterior_theta_param[idx][0] += mean_layer(self.h[-1][0])
                        #self.posterior_theta_param[idx][0][d] += mean_layer(self.h[-1][0][d])
                        #std_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: tf.nn.leaky_relu(x, alpha = 0.2))
                        #std_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: x)
                        #self.layers.append(std_layer)
                        #self.posterior_theta_param[idx][1][d] += std_layer(self.h[-1][1])
                        #self.posterior_theta_param[idx][1][d] += std_layer(self.h[-1][0][d])
                    
                    #pi_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: tf.nn.leaky_relu(x, alpha = 0.2))
                    pi_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: x)
                    self.layers.append(pi_layer)
                    self.posterior_theta_param[idx][2] += pi_layer(self.h[-1][2])
                    #self.posterior_theta_param[idx][2] += pi_layer(self.h[-1][1])
                    
                    #alpha_gam_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: tf.nn.softplus(x))
                    alpha_gam_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                    self.layers.append(alpha_gam_layer)
                    #self.posterior_theta_param[idx][3] += alpha_gam_layer(self.h[-1][3])
                    self.posterior_theta_param[idx][3] += alpha_gam_layer(self.h[-1][3])
                    #self.posterior_theta_param[idx][3] += alpha_gam_layer(self.h[-1][2])
                    
                    if FLAGS.directed == 1:
                        #alpha_del_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: tf.nn.softplus(x))
                        alpha_del_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                        self.layers.append(alpha_del_layer)
                        self.posterior_theta_param[idx][4] += alpha_del_layer(self.h[-1][4])
                        #self.posterior_theta_param[idx][3] += alpha_del_layer(self.h[-1][2])
                
                else:
                    """
                    phi_mean = []
                    with tf.compat.v1.variable_scope("phi_mean_" + str(idx), reuse = tf.compat.v1.AUTO_REUSE):
                        for d in range(self.latent_dim):
                            phi_mean.append(weight_variable_glorot(self.decoder_layers[idx - 1], decoder_layer, name = 'phi_mean_' + str(d) + '_' + str(idx)))
                            phi_mean[d] = tf.nn.softmax(phi_mean[d], axis = 0) # normalized
                        #phi = weight_variable_gamma(self.hidden[idx+1], self.hidden[idx])#, name='phi' + '_' + str(idx)) # (l+1)th non-negetive transiformation matrix: K^(l+1)×K^(l)
                
                    with tf.compat.v1.variable_scope("phi_pi_" + str(idx), reuse = tf.compat.v1.AUTO_REUSE): 
                        phi_pi = weight_variable_glorot(self.decoder_layers[idx - 1], decoder_layer, name = 'phi_pi_' + str(idx))
                        phi_pi = tf.nn.softmax(phi_pi, axis = 0) # normalized
                
                    with tf.compat.v1.variable_scope("phi_alpha_" + str(idx), reuse = tf.compat.v1.AUTO_REUSE): 
                        phi_alpha_gam = weight_variable_glorot(self.decoder_layers[idx - 1], decoder_layer, name = 'phi_alpha_gam_' + str(idx))
                        phi_alpha_gam = tf.nn.softmax(phi_alpha_gam, axis = 0) # normalized
                        if FLAGS.directed == 1:
                            phi_alpha_del = weight_variable_glorot(self.decoder_layers[idx - 1], decoder_layer, name = 'phi_alpha_del_' + str(idx))
                            phi_alpha_del = tf.nn.softmax(phi_alpha_del, axis = 0) # normalized
                        
                    for d in range(self.latent_dim):
                        self.posterior_theta_param[idx][0][d] += tf.matmul(self.theta[idx - 1][1][d], phi_mean[d]) # s^(l) + z^(l+1) * phi^(l+1)
                    self.posterior_theta_param[idx][2] += tf.matmul(self.theta[idx - 1][2], phi_pi)
                    self.posterior_theta_param[idx][3] += tf.matmul(self.theta[idx - 1][4], phi_alpha_gam)
                    if FLAGS.directed == 1:
                        self.posterior_theta_param[idx][4] += tf.matmul(self.theta[idx - 1][5], phi_alpha_del)
                    """
                    #mean_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: tf.nn.leaky_relu(x, alpha = 0.2))
                    mean_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: x)
                    self.layers.append(mean_layer)
                    #self.posterior_theta_param[idx][0] += mean_layer(self.theta[idx - 1][1])
                    self.posterior_theta_param[idx][0] += mean_layer(self.theta[idx - 1][0])
                    
                    #pi_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: tf.nn.leaky_relu(x, alpha = 0.2))
                    pi_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: x)
                    self.layers.append(pi_layer)
                    self.posterior_theta_param[idx][2] += pi_layer(self.theta[idx - 1][1])
                    
                    #alpha_gam_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: tf.nn.softplus(x))
                    alpha_gam_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                    self.layers.append(alpha_gam_layer)
                    #self.posterior_theta_param[idx][3] += alpha_gam_layer(self.theta[idx - 1][4])
                    self.posterior_theta_param[idx][3] += alpha_gam_layer(self.theta[idx - 1][3])
                    
                    if FLAGS.directed == 1:
                        #alpha_del_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: tf.nn.softplus(x))
                        alpha_del_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                        self.layers.append(alpha_del_layer)
                        #self.posterior_theta_param[idx][4] += alpha_del_layer(self.theta[idx - 1][5])
                        self.posterior_theta_param[idx][4] += alpha_del_layer(self.theta[idx - 1][4])
                
                # Processing top layer first
                
                #if FLAGS.directed == 1:
                    #v = sample_kumaraswamy(self.posterior_theta_param[idx][5], self.posterior_theta_param[idx][6]) # dim: N × K^L
                #else:
                    #v = sample_kumaraswamy(self.posterior_theta_param[idx][4], self.posterior_theta_param[idx][5]) # dim: N × K^L
                v = tf.constant(FLAGS.v0, shape = (self.n_samples, decoder_layer))
                pi_logit_prior = logit(tf.exp(tf.cumsum(tf.math.log(v + SMALL), axis = 1)))
                s_logit = sample_binconcrete(self.posterior_theta_param[idx][2], FLAGS.temp_post)
                s = tf.cond(pred = tf.equal(self.training, tf.constant(False)), true_fn = lambda: tf.round(tf.nn.sigmoid(s_logit)), false_fn = lambda: tf.nn.sigmoid(s_logit))
                
                z = sample_normal(self.posterior_theta_param[idx][0], self.posterior_theta_param[idx][1]) # N * K
                z = tf.multiply(s, z)
 
                alpha_gam_prior = tf.constant(1. / decoder_layer, tf.float32)
                #alpha_gam_prior = tf.reshape(1. / (self.outdegrees + 1), shape = (-1, 1))
                #alpha_gam_prior = tf.reshape(self.outdegrees / self.n_samples, [-1, 1])
                #gamma = sample_gamma(self.posterior_theta_param[idx][3], tf.constant(FLAGS.beta, tf.float32))
                gamma = sample_gamma(self.posterior_theta_param[idx][3], tf.constant(FLAGS.beta, tf.float32))
                #gamma = tf.random.gamma(shape = (self.n_samples, self.hidden[idx]), alpha = self.posterior_theta_param[-1][3], beta = FLAGS.beta)
                #gamma = self.n_samples * gamma / (tf.reduce_sum(gamma, axis = 0) + SMALL) # composing Gamma distribution (with same beta) as Dirichlet distribution
                gamma = gamma / (tf.reduce_sum(gamma, axis = 0) + SMALL) # composing Gamma distribution (with same beta) as Dirichlet distribution
                gamma = tf.multiply(s, gamma)
                #gamma = tf.compat.v1.check_numerics(gamma, 'gamma is nan')
                
                #self.prior_theta_param.append([pi_logit_prior, alpha_prior]) # arbritrary
            
                if FLAGS.directed == 1:
                    alpha_del_prior = tf.constant(1. / decoder_layer, tf.float32)
                    #alpha_del_prior = tf.reshape(1. / (self.indegrees + 1), shape = (-1, 1))
                    delta = sample_gamma(self.posterior_theta_param[idx][4], tf.constant(FLAGS.beta, tf.float32))
                    #delta = self.n_samples * delta / (tf.reduce_sum(delta, axis = 0) + SMALL) # composing Gamma distribution (with same beta) as Dirichlet distribution
                    delta = delta / (tf.reduce_sum(delta, axis = 0) + SMALL) # composing Gamma distribution (with same beta) as Dirichlet distribution
                    #delta = delta / (tf.reduce_sum(delta, axis = 0) + SMALL) # composing Gamma distribution (with same beta) as Dirichlet distribution
                    delta = tf.multiply(s, delta)
                    #delta = tf.compat.v1.check_numerics(delta, 'delta is nan')
                    
                    self.prior_theta_param.append([pi_logit_prior, alpha_gam_prior, alpha_del_prior]) # arbritrary
                    #self.theta.append([z, z_cluster, p, s, gamma, delta])
                    #self.theta.append([z, z_cluster, s_logit, s, gamma, delta])
                    self.theta.append([z, s_logit, s, gamma, delta])
                else:
                    self.prior_theta_param.append([pi_logit_prior, alpha_gam_prior]) # arbritrary
                    #self.theta.append([z, z_cluster, p, s, gamma])
                    #self.theta.append([z, z_cluster, s_logit, s, gamma])
                    self.theta.append([z, s_logit, s, gamma])
                    
                    #self.reg_phi += tf.nn.l2_loss(phi)
                    #Will be messed up if you take mc-samples!!!
                    #self.phi.append(phi)
               
                #False deactivates :)
                if(False and FLAGS.features==1 and FLAGS.reconstruct_x==1):
                    #feature decoder (starting from L-1th layer)
                    if idx == 0:
                        #last layer remains real
                        act_x = lambda x: x
                    else:
                        act_x = tf.nn.relu
                    x_h = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = act_x, reuse_name = 'conv_weight_' + str(idx+1), reuse = True, transpose = True)(x_h)

            # reverse
            #self.theta = theta[::-1] # z^1, z^2, ..., z^L       dim: L × 1 × d × N × K
            #self.theta_concat = tf.concat(self.theta, axis=1) # dim: N × L*K
            #print(self.theta_concat.get_shape().as_list())

            #self.prior_theta_param = self.prior_theta_param[::-1]
            
            #output_layer = LSMDecoder(input_dim = self.hidden[0], logging = self.logging)
            output_layer = LSMDecoder(input_dim = self.decoder_layers[-1], num_nodes = self.n_samples, act = lambda x: tf.nn.sigmoid(x), logging = self.logging)
            #self.last_layer = output_layer

            #self.reconstructions_list.append(weight_layer(transformed_theta))
            
            #epsilon = tf.constant(10e-10)
            #self.poisson_rate = tf.clip_by_value(self.reconstructions_list[0], epsilon, 10e10)
            #self.rate = 1 - tf.exp(-self.poisson_rate)
            
            #self.poisson_rate = output_layer(self.theta[0], act = lambda x: x) #self.reconstructions_list[s]
            #self.poisson_rate, self.z_decoder = output_layer(self.theta[-1])
            self.reconstructions, self.theta_decoder = output_layer(self.theta[-1])
            #self.poisson_rate = output_layer(self.theta[-1])
            
            #if FLAGS.data_type == 'binary':
            #    self.clipped_logit = tf.clip_by_value(self.poisson_rate,0.0,1.0)  #tf.log(self.poisson_rate) #tf.log(self.rate)
            #else:
            #    self.clipped_logit = self.poisson_rate

            #self.reconstructions = self.clipped_logit
            
            self.theta_list.append(self.theta)
            self.reconstructions_list.append(self.reconstructions)
            self.posterior_theta_param_list.append(self.posterior_theta_param)
            self.prior_theta_param_list.append(self.prior_theta_param)
        ###############################################################
        
        if(FLAGS.features==1 and FLAGS.reconstruct_x==1):
            #last transform for Decoder
            x_h = FullConnection(input_dim = self.hidden[0], output_dim = self.input_dim, reuse_name = 'conv_weight_'+str(0), reuse = True, transpose = True)(self.theta[0])
            #No non-linearity for last layer?
            self.x_recon = tf.reshape(x_h, [-1])

        if(FLAGS.semisup_train == 1):
            #can use final or prefinal layer or last d
            """
            self.z =  GraphConvolution(input_dim=self.hidden[-1],
                                     output_dim=self.num_classes,
                                     adj=self.adj,
                                     act = tf.nn.relu, #tf.nn.relu,
                                     dropout=self.dropout,
                                     name = "conv_weight_classify_"+str(idx),
                                     logging=self.logging)(d)
            """

            def weibull_mean(params):
                k = params[0]
                l = params[1]

                return l * tf.exp(tf.math.lgamma(1 + 1/k))

            classification_layer = FullConnection(input_dim = self.hidden[0], output_dim = self.num_classes, name = 'semisup_weight',dropout=0.)
            self.z = classification_layer(self.theta[0])

        #self.lambda_mat = self.last_layer.get_weight_matrix()

