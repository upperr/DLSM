from initializations import weight_variable_glorot
import tensorflow as tf
from utils import distance_scaled_by_gam, distance_scaled_by_del

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
SMALL = 1e-16

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.compat.v1.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

class FullConnection(Layer):
    def __init__(self, input_dim, output_dim, act, dropout = 0., reuse_name = '', **kwargs):
        super().__init__(**kwargs)
        
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name = "weights")
        
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['bias'] = tf.Variable(tf.zeros((output_dim)), name = "bias")
        
        self.act = act
        self.dropout = dropout

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, self.dropout)
        x = tf.matmul(x, self.vars['weights']) + self.vars['bias']
        
        output = self.act(x)
        return output

class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, num_nodes, act = tf.nn.relu, dropout = 0., **kwargs):
        super().__init__(**kwargs)
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
            #self.vars['bias'] = tf.Variable(tf.zeros((output_dim)), name="bias")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, self.dropout)
        x = tf.matmul(x, self.vars['weights']) #+ self.vars['bias']
        x = tf.sparse.sparse_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs
    
class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, num_nodes, features_nonzero, act = tf.nn.relu, dropout = 0., **kwargs):
        super().__init__(**kwargs)
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
            #self.vars['bias'] = tf.Variable(tf.zeros((output_dim)), name="bias")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
        x = tf.sparse.sparse_dense_matmul(x, self.vars['weights']) #+ self.vars['bias']
        x = tf.sparse.sparse_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

class InnerProductDecoder(Layer):
    """Decoder model layer for DLSM-IP."""  
    def __init__(self, input_dim, num_nodes, dropout = 0., act = tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.input_dim = input_dim
        self.n_samples = num_nodes
        self.latent_dim = FLAGS.latent_dim
        
        with tf.compat.v1.variable_scope(self.name + '_vars_z'):
            self.vars['transform_z'] = weight_variable_glorot(input_dim, self.latent_dim, name = "transform_z")
            #self.vars['transform_z'] = tf.nn.softmax(self.vars['transform_z'], axis = 0)
        
        with tf.compat.v1.variable_scope(self.name + '_vars_gam'):
            self.vars['transform_gam'] = weight_variable_glorot(input_dim, self.latent_dim, name = "transform_gam")
            #self.vars['transform_gam'] = tf.nn.softmax(self.vars['transform_gam'], axis = 0) # normalized
            
        if FLAGS.directed == 1:
            with tf.compat.v1.variable_scope(self.name + '_vars_del'):
                self.vars['transform_del'] = weight_variable_glorot(input_dim, self.latent_dim, name = "transform_del")
                #self.vars['transform_del'] = tf.nn.softmax(self.vars['transform_del'], axis = 0) # normalized

            #with tf.compat.v1.variable_scope(self.name + '_vars_weights'):
                #self.vars['weight_gam'] = tf.Variable(0.5, name = "weight_gam")
                #self.vars['weight_del'] = tf.Variable(0.5, name = "weight_del")
        #else:
            #with tf.compat.v1.variable_scope(self.name + '_vars_weights'):
                #self.vars['weight_gam'] = tf.Variable(0.5, name = "weight_gam")

    def _call(self, inputs):
        
        z = tf.nn.dropout(inputs[0], self.dropout)
        z_decoder = tf.matmul(z, self.vars['transform_z'])
        gamma = inputs[3]
        gamma_decoder = tf.matmul(gamma, self.vars['transform_gam'])# * self.n_samples
        
        x1 = tf.multiply(z_decoder, gamma_decoder)
        
        if FLAGS.directed == 1:
            delta = inputs[4]
            delta_decoder = tf.matmul(delta, self.vars['transform_del'])# * self.n_samples
            
            x2 = tf.multiply(z_decoder, delta_decoder)
        else:
            delta_decoder = gamma_decoder
            x2 = x1
        
        #x = tf.matmul(self.vars['weight_gam'] * x1, self.vars['weight_del'] * tf.transpose(x2))
        x = tf.matmul(x1, tf.transpose(x2))
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs, (z_decoder, gamma_decoder, delta_decoder)
    
class LSMDecoder(Layer):
    """Decoder model layer for DLSM."""
    def __init__(self, input_dim, num_nodes, act = tf.nn.sigmoid, dropout = 0., **kwargs):
        super().__init__(**kwargs)
                
        self.act = act
        self.dropout = dropout
        self.input_dim = input_dim
        self.n_samples = num_nodes
        self.latent_dim = FLAGS.latent_dim
            
        with tf.compat.v1.variable_scope(self.name + '_vars_bias'):
            self.vars['bias'] = tf.Variable(1., name = "bias")
        
        with tf.compat.v1.variable_scope(self.name + '_vars_z'):
            self.vars['transform_z'] = weight_variable_glorot(input_dim, self.latent_dim, name = "transform_z")
            self.vars['transform_z'] = tf.nn.softmax(self.vars['transform_z'], axis = 0)
        
        with tf.compat.v1.variable_scope(self.name + '_vars_gam'):
            self.vars['transform_gam'] = weight_variable_glorot(input_dim, self.latent_dim, name = "transform_gam")
            self.vars['transform_gam'] = tf.nn.softmax(self.vars['transform_gam'], axis = 0) # normalized
            
        if FLAGS.directed == 1:
            with tf.compat.v1.variable_scope(self.name + '_vars_del'):
                self.vars['transform_del'] = weight_variable_glorot(input_dim, self.latent_dim, name = "transform_del")
                self.vars['transform_del'] = tf.nn.softmax(self.vars['transform_del'], axis = 0) # normalized

            with tf.compat.v1.variable_scope(self.name + '_vars_weights'):
                self.vars['weight_gam'] = tf.Variable(0.5, name = "weight_gam")
                self.vars['weight_del'] = tf.Variable(0.5, name = "weight_del")
        else:
            with tf.compat.v1.variable_scope(self.name + '_vars_weights'):
                self.vars['weight_gam'] = tf.Variable(0.5, name = "weight_gam")

    def _call(self, inputs):
        #z = tf.nn.dropout(inputs[0], self.dropout)
        z = inputs[0]
        z_decoder = tf.matmul(z, self.vars['transform_z'])
        
        gamma = inputs[3]
        gamma_decoder = tf.matmul(gamma, self.vars['transform_gam'])
        dist_gam = self.n_samples * distance_scaled_by_gam(z_decoder, gamma_decoder + SMALL)
        
        if FLAGS.directed == 1:
            delta = inputs[4]
            delta_decoder = tf.matmul(delta, self.vars['transform_del'])
            dist_del = self.n_samples * distance_scaled_by_del(z_decoder, delta_decoder + SMALL)
            
            x = self.vars['bias'] - self.vars['weight_gam'] * dist_gam - self.vars['weight_del'] * dist_del
        else:
            delta_decoder = gamma_decoder
            x = self.vars['bias'] - self.vars['weight_gam'] * (dist_gam + tf.transpose(dist_gam))

        x = tf.reshape(x, [-1])
        output = self.act(x)

        return output, (z_decoder, gamma_decoder, delta_decoder)
    
    def get_weight_matrix(self):
        W = tf.eye(self.input_dim)
        return W
