from layers import FullConnection, GraphConvolution, GraphConvolutionSparse, LSMDecoder, InnerProductDecoder
import tensorflow as tf
from utils import *

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
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, mc_samples = 1, **kwargs):
        super().__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.training = placeholders['is_training']

        self.encoder_layers = [int(x) for x in FLAGS.encoder.split('_')]
        self.decoder_layers = [int(x) for x in FLAGS.decoder.split('_')]
        self.num_encoder_layers = len(self.encoder_layers)
        self.num_decoder_layers = len(self.decoder_layers)
        
        self.prior_theta_param = []
        self.posterior_theta_param = []
        self.S = mc_samples #No. of MC samples
        
        self.build()
    
    def _build(self):

        print('Build Dynamic Network....')

        self.posterior_theta_param = []
        self.h = []
        self.layers = []

        # Upward Inference Pass
        for idx, encoder_layer in enumerate(self.encoder_layers):

            #act = lambda a: tf.nn.leaky_relu(a, alpha=0.2)
            
            #This selection is questionable. May not be much of effect in reality
            act = tf.nn.sigmoid
            
            if idx == 0:
                
                gc = GraphConvolutionSparse(input_dim = self.input_dim,
                                            output_dim = encoder_layer,
                                            adj = self.adj,
                                            num_nodes = self.n_samples,
                                            features_nonzero = self.features_nonzero,
                                            act = lambda x: x,
                                            name = "conv_weight_input_" + str(idx),
                                            dropout = self.dropout,
                                            logging = self.logging)
                h = gc(self.inputs)
                
                if FLAGS.directed == 1:
                    self.h.append([h, h, h, h, h])
                else:
                    self.h.append([h, h, h, h])

            else:
                gc_mean = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                     output_dim = encoder_layer,
                                     adj = self.adj,
                                     num_nodes = self.n_samples,
                                     act = act,
                                     dropout = self.dropout,
                                     name = "conv_weight_mean_" + str(idx),
                                     logging = self.logging)
                h_mean = gc_mean(self.h[-1][0])
                
                gc_std = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                     output_dim = encoder_layer,
                                     adj = self.adj,
                                     num_nodes = self.n_samples,
                                     act = act,
                                     dropout = self.dropout,
                                     name = "conv_weight_std_" + str(idx),
                                     logging = self.logging)
                h_std = gc_std(self.h[-1][1])
                
                gc_pi = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                     output_dim = encoder_layer,
                                     adj = self.adj,
                                     num_nodes = self.n_samples,
                                     act = act,
                                     dropout = self.dropout,
                                     name = "conv_weight_pi_" + str(idx),
                                     logging = self.logging)
                h_pi = gc_pi(self.h[-1][2])
                
                gc_alpha_gam = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                     output_dim = encoder_layer,
                                     adj = self.adj,
                                     num_nodes = self.n_samples,
                                     act = act,
                                     dropout = self.dropout,
                                     name = "conv_weight_alpha_gam_" + str(idx),
                                     logging = self.logging)
                h_alpha_gam = gc_alpha_gam(self.h[-1][3])
                
                if FLAGS.directed == 1:
                    gc_alpha_del = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                     output_dim = encoder_layer,
                                     adj = self.adj,
                                     num_nodes = self.n_samples,
                                     act = act,
                                     dropout = self.dropout,
                                     name = "conv_weight_alpha_del_" + str(idx),
                                     logging = self.logging)
                    h_alpha_del = gc_alpha_del(self.h[-1][4])
                    
                    self.h.append([h_mean, h_std, h_pi, h_alpha_gam, h_alpha_del])
                    
                else:
                    self.h.append([h_mean, h_std, h_pi, h_alpha_gam])

            # get Theta parameters
                
                mean_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: x)
                z_mean = mean_layer(self.h[-1][0])
                
                std_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: x)
                z_std = std_layer(self.h[-1][1])
                
                pi_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: x)
                pi_logit = pi_layer(self.h[-1][2])
                
                alpha_gam_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: tf.nn.softplus(x))
                alpha_gam = alpha_gam_layer(self.h[-1][3])
            
                if FLAGS.directed == 1:
                    alpha_del_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: tf.nn.softplus(x))
                    alpha_del = alpha_del_layer(self.h[-1][4])
            
                    self.posterior_theta_param.append([z_mean, z_std, pi_logit, alpha_gam, alpha_del])
                else:
                    self.posterior_theta_param.append([z_mean, z_std, pi_logit, alpha_gam])

        # Downward Inference pass

        self.theta_list = []      
        self.reconstructions_list = []
        self.posterior_theta_param_list = []
        self.prior_theta_param_list = []
        
        ###########################################################################
        #Take multiple MC samples
        for k in range(self.S):
            # Refresh
            
            # Downward Inference Pass
            self.theta = []
            self.prior_theta_param = []
            
            # Downward Inference Pass
            for idx, decoder_layer in enumerate(self.decoder_layers): # l = 1, 2, ..., L-1
                
                if idx == 0:

                    mean_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: x)
                    self.posterior_theta_param[idx][0] += mean_layer(self.h[-1][0])
                    
                    pi_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: x)
                    self.posterior_theta_param[idx][2] += pi_layer(self.h[-1][2])
                    
                    alpha_gam_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                    self.posterior_theta_param[idx][3] += alpha_gam_layer(self.h[-1][3])
                    
                    if FLAGS.directed == 1:
                        alpha_del_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                        self.posterior_theta_param[idx][4] += alpha_del_layer(self.h[-1][4])
                
                else:
                    mean_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: x)
                    self.posterior_theta_param[idx][0] += mean_layer(self.theta[idx - 1][0])
                    
                    pi_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: x)
                    self.posterior_theta_param[idx][2] += pi_layer(self.theta[idx - 1][1])
                    
                    alpha_gam_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                    self.posterior_theta_param[idx][3] += alpha_gam_layer(self.theta[idx - 1][3])
                    
                    if FLAGS.directed == 1:
                        alpha_del_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                        self.posterior_theta_param[idx][4] += alpha_del_layer(self.theta[idx - 1][4])
                
                # Processing top layer first

                v = tf.constant(FLAGS.v0, shape = (self.n_samples, decoder_layer))
                pi_logit_prior = logit(tf.exp(tf.cumsum(tf.math.log(v + SMALL), axis = 1)))
                s_logit = sample_binconcrete(self.posterior_theta_param[idx][2], FLAGS.temp_post)
                s = tf.cond(pred = tf.equal(self.training, tf.constant(False)), true_fn = lambda: tf.round(tf.nn.sigmoid(s_logit)), false_fn = lambda: tf.nn.sigmoid(s_logit))
                
                z = sample_normal(self.posterior_theta_param[idx][0], self.posterior_theta_param[idx][1])
                z = tf.multiply(s, z)
 
                alpha_gam_prior = tf.constant(1. / decoder_layer, tf.float32)
                #alpha_gam_prior = tf.reshape(1. / (self.outdegrees + 1), shape = (-1, 1))
                gamma = sample_gamma(self.posterior_theta_param[idx][3], tf.constant(FLAGS.beta, tf.float32))
                #gamma = self.n_samples * gamma / (tf.reduce_sum(gamma, axis = 0) + SMALL) # composing Gamma distribution (with same beta) as Dirichlet distribution
                gamma = gamma / (tf.reduce_sum(gamma, axis = 0) + SMALL) # composing Gamma distribution (with same beta) as Dirichlet distribution
                gamma = tf.multiply(s, gamma)
            
                if FLAGS.directed == 1:
                    alpha_del_prior = tf.constant(1. / decoder_layer, tf.float32)
                    #alpha_del_prior = tf.reshape(1. / (self.indegrees + 1), shape = (-1, 1))
                    delta = sample_gamma(self.posterior_theta_param[idx][4], tf.constant(FLAGS.beta, tf.float32))
                    #delta = self.n_samples * delta / (tf.reduce_sum(delta, axis = 0) + SMALL) # composing Gamma distribution (with same beta) as Dirichlet distribution
                    delta = delta / (tf.reduce_sum(delta, axis = 0) + SMALL) # composing Gamma distribution (with same beta) as Dirichlet distribution
                    delta = tf.multiply(s, delta)
                    
                    self.prior_theta_param.append([pi_logit_prior, alpha_gam_prior, alpha_del_prior])
                    self.theta.append([z, s_logit, s, gamma, delta])
                else:
                    self.prior_theta_param.append([pi_logit_prior, alpha_gam_prior])
                    self.theta.append([z, s_logit, s, gamma])

            output_layer = LSMDecoder(input_dim = self.decoder_layers[-1], num_nodes = self.n_samples, act = lambda x: tf.nn.sigmoid(x), logging = self.logging)
            
            self.reconstructions, self.theta_decoder = output_layer(self.theta[-1])
            
            self.theta_list.append(self.theta)
            self.reconstructions_list.append(self.reconstructions)
            self.posterior_theta_param_list.append(self.posterior_theta_param)
            self.prior_theta_param_list.append(self.prior_theta_param)
        ###############################################################

class DLSM_IP(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, mc_samples = 1, **kwargs):
        super().__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.training = placeholders['is_training']

        self.encoder_layers = [int(x) for x in FLAGS.encoder.split('_')]
        self.decoder_layers = [int(x) for x in FLAGS.decoder.split('_')]
        self.num_encoder_layers = len(self.encoder_layers)
        self.num_decoder_layers = len(self.decoder_layers)
        
        self.prior_theta_param = []
        self.posterior_theta_param = []
        self.S = mc_samples #No. of MC samples
        
        self.build()
    
    def _build(self):

        print('Build Dynamic Network....')

        self.posterior_theta_param = []
        self.h = []
        self.layers = []

        # Upward Inference Pass
        for idx, encoder_layer in enumerate(self.encoder_layers):

            #act = lambda a: tf.nn.leaky_relu(a, alpha=0.2)
            
            #This selection is questionable. May not be much of effect in reality
            act = tf.nn.sigmoid
            
            if idx == 0:
                
                gc = GraphConvolutionSparse(input_dim = self.input_dim,
                                            output_dim = encoder_layer,
                                            adj = self.adj,
                                            num_nodes = self.n_samples,
                                            features_nonzero = self.features_nonzero,
                                            act = lambda x: x,
                                            name = "conv_weight_input_" + str(idx),
                                            dropout = self.dropout,
                                            logging = self.logging)
                h = gc(self.inputs)
                
                if FLAGS.directed == 1:
                    self.h.append([h, h, h, h, h])
                else:
                    self.h.append([h, h, h, h])

            else:
                gc_mean = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                     output_dim = encoder_layer,
                                     adj = self.adj,
                                     num_nodes = self.n_samples,
                                     act = act,
                                     dropout = self.dropout,
                                     name = "conv_weight_mean_" + str(idx),
                                     logging = self.logging)
                h_mean = gc_mean(self.h[-1][0])
                
                gc_std = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                     output_dim = encoder_layer,
                                     adj = self.adj,
                                     num_nodes = self.n_samples,
                                     act = act,
                                     dropout = self.dropout,
                                     name = "conv_weight_std_" + str(idx),
                                     logging = self.logging)
                h_std = gc_std(self.h[-1][1])
                
                gc_pi = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                     output_dim = encoder_layer,
                                     adj = self.adj,
                                     num_nodes = self.n_samples,
                                     act = act,
                                     dropout = self.dropout,
                                     name = "conv_weight_pi_" + str(idx),
                                     logging = self.logging)
                h_pi = gc_pi(self.h[-1][2])
                
                gc_alpha_gam = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                     output_dim = encoder_layer,
                                     adj = self.adj,
                                     num_nodes = self.n_samples,
                                     act = act,
                                     dropout = self.dropout,
                                     name = "conv_weight_alpha_gam_" + str(idx),
                                     logging = self.logging)
                h_alpha_gam = gc_alpha_gam(self.h[-1][3])
                
                if FLAGS.directed == 1:
                    gc_alpha_del = GraphConvolution(input_dim = self.encoder_layers[idx - 1],
                                     output_dim = encoder_layer,
                                     adj = self.adj,
                                     num_nodes = self.n_samples,
                                     act = act,
                                     dropout = self.dropout,
                                     name = "conv_weight_alpha_del_" + str(idx),
                                     logging = self.logging)
                    h_alpha_del = gc_alpha_del(self.h[-1][4])
                    
                    self.h.append([h_mean, h_std, h_pi, h_alpha_gam, h_alpha_del])
                    
                else:
                    self.h.append([h_mean, h_std, h_pi, h_alpha_gam])

            # get Theta parameters
                
                mean_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: x)
                z_mean = mean_layer(self.h[-1][0])
                
                std_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: x)
                z_std = std_layer(self.h[-1][1])
                
                pi_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: x)
                pi_logit = pi_layer(self.h[-1][2])
                
                alpha_gam_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: tf.nn.softplus(x))
                alpha_gam = alpha_gam_layer(self.h[-1][3])
            
                if FLAGS.directed == 1:
                    alpha_del_layer = FullConnection(input_dim = encoder_layer, output_dim = self.decoder_layers[idx - 1], act = lambda x: tf.nn.softplus(x))
                    alpha_del = alpha_del_layer(self.h[-1][4])
            
                    self.posterior_theta_param.append([z_mean, z_std, pi_logit, alpha_gam, alpha_del])
                else:
                    self.posterior_theta_param.append([z_mean, z_std, pi_logit, alpha_gam])

        # Downward Inference pass

        self.theta_list = []      
        self.reconstructions_list = []
        self.posterior_theta_param_list = []
        self.prior_theta_param_list = []
        
        ###########################################################################
        #Take multiple MC samples
        for k in range(self.S):
            # Refresh
            
            # Downward Inference Pass
            self.theta = []
            self.prior_theta_param = []
            
            # Downward Inference Pass
            for idx, decoder_layer in enumerate(self.decoder_layers): # l = 1, 2, ..., L-1
                
                if idx == 0:

                    mean_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: x)
                    self.posterior_theta_param[idx][0] += mean_layer(self.h[-1][0])
                    
                    pi_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: x)
                    self.posterior_theta_param[idx][2] += pi_layer(self.h[-1][2])
                    
                    alpha_gam_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                    self.posterior_theta_param[idx][3] += alpha_gam_layer(self.h[-1][3])
                    
                    if FLAGS.directed == 1:
                        alpha_del_layer = FullConnection(input_dim = self.encoder_layers[-1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                        self.posterior_theta_param[idx][4] += alpha_del_layer(self.h[-1][4])
                
                else:
                    mean_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: x)
                    self.posterior_theta_param[idx][0] += mean_layer(self.theta[idx - 1][0])
                    
                    pi_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: x)
                    self.posterior_theta_param[idx][2] += pi_layer(self.theta[idx - 1][1])
                    
                    alpha_gam_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                    self.posterior_theta_param[idx][3] += alpha_gam_layer(self.theta[idx - 1][3])
                    
                    if FLAGS.directed == 1:
                        alpha_del_layer = FullConnection(input_dim = self.decoder_layers[idx - 1], output_dim = decoder_layer, act = lambda x: tf.nn.relu(x))
                        self.posterior_theta_param[idx][4] += alpha_del_layer(self.theta[idx - 1][4])
                
                # Processing top layer first

                v = tf.constant(FLAGS.v0, shape = (self.n_samples, decoder_layer))
                pi_logit_prior = logit(tf.exp(tf.cumsum(tf.math.log(v + SMALL), axis = 1)))
                s_logit = sample_binconcrete(self.posterior_theta_param[idx][2], FLAGS.temp_post)
                s = tf.cond(pred = tf.equal(self.training, tf.constant(False)), true_fn = lambda: tf.round(tf.nn.sigmoid(s_logit)), false_fn = lambda: tf.nn.sigmoid(s_logit))
                
                z = sample_normal(self.posterior_theta_param[idx][0], self.posterior_theta_param[idx][1])
                z = tf.multiply(s, z)
 
                alpha_gam_prior = tf.constant(1. / decoder_layer, tf.float32)
                #alpha_gam_prior = tf.reshape(1. / (self.outdegrees + 1), shape = (-1, 1))
                gamma = sample_gamma(self.posterior_theta_param[idx][3], tf.constant(FLAGS.beta, tf.float32))
                #gamma = gamma / (tf.reduce_sum(gamma, axis = 0) + SMALL) # composing Gamma distribution (with same beta) as Dirichlet distribution
                gamma = tf.multiply(s, gamma)
            
                if FLAGS.directed == 1:
                    alpha_del_prior = tf.constant(1. / decoder_layer, tf.float32)
                    #alpha_del_prior = tf.reshape(1. / (self.indegrees + 1), shape = (-1, 1))
                    delta = sample_gamma(self.posterior_theta_param[idx][4], tf.constant(FLAGS.beta, tf.float32))
                    #delta = delta / (tf.reduce_sum(delta, axis = 0) + SMALL) # composing Gamma distribution (with same beta) as Dirichlet distribution
                    delta = tf.multiply(s, delta)
                    
                    self.prior_theta_param.append([pi_logit_prior, alpha_gam_prior, alpha_del_prior])
                    self.theta.append([z, s_logit, s, gamma, delta])
                else:
                    self.prior_theta_param.append([pi_logit_prior, alpha_gam_prior])
                    self.theta.append([z, s_logit, s, gamma])

            output_layer = InnerProductDecoder(input_dim = self.decoder_layers[-1], num_nodes = self.n_samples, act = lambda x: x, logging = self.logging)
            
            self.reconstructions, self.theta_decoder = output_layer(self.theta[-1])
            
            self.theta_list.append(self.theta)
            self.reconstructions_list.append(self.reconstructions)
            self.posterior_theta_param_list.append(self.posterior_theta_param)
            self.prior_theta_param_list.append(self.prior_theta_param)