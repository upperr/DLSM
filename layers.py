from initializations import *
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

class LinearLayer(Layer):
    def __init__(self, input_dim, output_dim, dropout=0., reuse_name='', reuse=False, transpose = False, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        # reuse : for weight reuse -- tied weights
        # transpose : if reuse => for decoder part
        
        if(reuse):
            #reuse conv weights 
            with tf.compat.v1.variable_scope(reuse_name + '_vars', reuse = True):
                self.vars['weights'] = tf.compat.v1.get_variable('weights')

                if(transpose):
                    self.vars['weights'] = tf.transpose(a=self.vars['weights'])
                print(self.vars['weights'].name)
        
        else:
            with tf.compat.v1.variable_scope(self.name + '_vars'):
                self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['bias'] = tf.Variable(tf.zeros((output_dim)), name="bias")
        self.dropout = dropout

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1 - (1-self.dropout))
        output = tf.matmul(x, self.vars['weights']) + self.vars['bias']
        return output

class FullConnection(Layer):
    def __init__(self, input_dim, output_dim, act, dropout=0., reuse_name='', reuse=False, transpose = False, **kwargs):
        #super(LinearLayer, self).__init__(**kwargs)
        super().__init__(**kwargs)

        # reuse : for weight reuse -- tied weights
        # transpose : if reuse => for decoder part
        
        if(reuse):
            #reuse conv weights 
            with tf.compat.v1.variable_scope(reuse_name + '_vars', reuse = tf.compat.v1.AUTO_REUSE):
                self.vars['weights'] = tf.compat.v1.get_variable('weights')

                if(transpose):
                    self.vars['weights'] = tf.transpose(a=self.vars['weights'])
                print(self.vars['weights'].name)
        
        else:
            with tf.compat.v1.variable_scope(self.name + '_vars'):
                self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['bias'] = tf.Variable(tf.zeros((output_dim)), name="bias")
        
        self.act = act
        self.dropout = dropout

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, self.dropout)
        x = tf.matmul(x, self.vars['weights']) + self.vars['bias']
        
        output = self.act(x)
        return output
        
class SparseLinearLayer(Layer):
    def __init__(self, input_dim, output_dim, features_nonzero, dropout=0., reuse_name='', reuse=False, transpose = False, **kwargs):
        super(SparseLinearLayer, self).__init__(**kwargs)

        # reuse : for weight reuse -- tied weights
        # transpose : if reuse => for decoder part

        if(reuse):
            #reuse conv weights 
            with tf.compat.v1.variable_scope(reuse_name + '_vars', reuse = True):
                self.vars['weights'] = tf.compat.v1.get_variable('weights')

                if(transpose):
                    self.vars['weights'] = tf.transpose(a=self.vars['weights'])
                print(self.vars['weights'].name)
        
        else:
            with tf.compat.v1.variable_scope(self.name + '_vars'):
                self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['bias'] = tf.Variable(tf.zeros((output_dim)), name="bias")
        self.dropout = dropout
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        output = tf.sparse.sparse_dense_matmul(x, self.vars['weights']) + self.vars['bias']
        return output

class GraphConvolution1(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, outdegrees, indegrees, edges, num_nodes, act = tf.nn.relu, dropout=0., **kwargs):
        #super(GraphConvolution, self).__init__(**kwargs)
        super().__init__(**kwargs)
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name = "weights")
            self.vars['bias'] = tf.Variable(tf.zeros((output_dim)), name = "bias")
            
        with tf.compat.v1.variable_scope(self.name + '_attention_vars'):
            self.vars['phi_attention'] = weight_variable_glorot(2 * output_dim, 1, name = 'weights_attention')
            self.vars['bias_attention'] = tf.Variable(tf.zeros((1)), name = "bias_attention")
        """
        self.gating_weight_dim = int(FLAGS.gating_weight)
        with tf.compat.v1.variable_scope(self.name + '_gate_vars'):
            self.vars['gating_weight1'] = weight_variable_glorot(input_dim, self.gating_weight_dim, name = 'gating_weight1')

            self.vars['gating_weight2'] = weight_variable_glorot(self.gating_weight_dim, output_dim, name = 'gating_weight2')
                                                
            self.vars['gating_bias1'] = weight_variable_glorot(1, self.gating_weight_dim, name = 'gating_bias1')

            self.vars['gating_bias2'] = tf.Variable(tf.zeros((1)), name = "gating_bias2")
        """
        if FLAGS.directed == 1:
            with tf.compat.v1.variable_scope(self.name + '_degree_vars'):
                self.vars['weight_degree_out'] = weight_variable_glorot(1, 1, name = "weight_degree_out")
                self.vars['weight_degree_in'] = weight_variable_glorot(1, 1, name = "weight_degree_in")
                #self.vars['bias_degree'] = tf.Variable(tf.zeros((1)), name = "bias_degree")
            
            with tf.compat.v1.variable_scope(self.name + '_merge_vars'):
                self.vars['weight_self'] = weight_variable_glorot(1, 1, name = 'weight_self')
                self.vars['weight_neighbor_out'] = weight_variable_glorot(1, 1, name = 'weight_neighbor_out')
                self.vars['weight_neighbor_in'] = weight_variable_glorot(1, 1, name = 'weight_neighbor_in')
                #self.vars['bias_merge'] = tf.Variable(tf.zeros((1)), name = "bias_merge")
        else:
            with tf.compat.v1.variable_scope(self.name + '_degree_vars'):
                self.vars['weight_degree'] = weight_variable_glorot(1, 1, name = "weight_degree")
                #self.vars['bias_degree'] = tf.Variable(tf.zeros((1)), name = "bias_degree")
            
            with tf.compat.v1.variable_scope(self.name + '_merge_vars'):
                self.vars['weight_self'] = weight_variable_glorot(1, 1, name = 'weight_self')
                self.vars['weight_neighbor'] = weight_variable_glorot(1, 1, name = 'weight_neighbor')
                #self.vars['bias_merge'] = tf.Variable(tf.zeros((1)), name = "bias_merge")
            
        self.output_dim = output_dim
        self.n_samples = num_nodes
        self.act = act
        self.outdegrees = outdegrees
        self.indegrees = indegrees
        self.edges = edges
        #self.ind_sender = ind_sender
        #self.ind_receiver = ind_receiver
        self.dropout = dropout
        self.adj = adj

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, self.dropout)
        x_transformed = tf.matmul(x, self.vars['weights']) + self.vars['bias']
        x_transformed = tf.sparse.sparse_dense_matmul(self.adj, x_transformed)
        
        if FLAGS.directed == 1:
            gate_degree = tf.nn.sigmoid(self.vars['weight_degree_out'] * tf.reshape(self.outdegrees, [-1, 1]) + self.vars['weight_degree_in'] * tf.reshape(self.indegrees, [-1, 1])) #+ self.vars['bias_degree']) # dim: N × 1
        else:
            gate_degree = tf.nn.sigmoid(self.vars['weight_degree'] * tf.reshape(self.outdegrees, [-1, 1])) #+ self.vars['bias_degree']) # dim: N × 1
        
        #gate = tf.reshape(tf.nn.elu(tf.matmul(
        #                            tf.nn.elu(tf.matmul(
        #                                tf.reshape(x,[-1,1]), self.vars['gating_weight1']) + self.vars['gating_bias1']),
        #                            self.vars['gating_weight2']) + self.vars['gating_bias2']), [-1, 1])
        x_filtered = tf.multiply(x_transformed, gate_degree) # dim: N × K^(l + 1)
        #x_filtered = x_transformed # dim: N × K^(l + 1)
        #x_filtered = tf.multiply(x_transformed, gate) # dim: N × K^(l + 1)

        #get self and neighbor representations for attention mechanism
        [ind_sender, ind_receiver] = tf.split(self.edges, num_or_size_splits = 2, axis = 1) # indices of senders (row) and recievers (column)
        #repre_neighbor = tf.nn.embedding_lookup(params = x_transformed, ids = tf.reduce_sum(self.ind_receiver, axis = 1)) # dim: edge_number × K^(l + 1)
        repre_neighbor = tf.nn.embedding_lookup(params = x_transformed, ids = tf.reduce_sum(ind_receiver, axis = 1)) # dim: edge_number × K^(l + 1)
        #repre_self = tf.nn.embedding_lookup(params = x_transformed, ids = tf.reduce_sum(self.ind_sender, axis = 1)) # dim: edge_num × K^(l + 1)
        repre_self = tf.nn.embedding_lookup(params = x_transformed, ids = tf.reduce_sum(ind_sender, axis = 1)) # dim: edge_num × K^(l + 1)
        repre_concat = tf.concat([repre_neighbor, repre_self], axis = 1) # dim: edge_num × 2*K^(l + 1)

        # calculate attention weight
        attention_val = tf.reshape(tf.nn.leaky_relu(tf.matmul(repre_concat, self.vars['phi_attention']) + self.vars['bias_attention'], alpha = 0.2), [-1]) # dim: edge_num
        #edges_sender2receiver = tf.concat([self.ind_sender, self.ind_receiver], axis = 1)
        #attention_out = tf.compat.v1.SparseTensorValue(indices = edges_sender2receiver, values = attention_val, dense_shape = (self.n_samples, self.n_samples)) # dim: N × N
        attention_out = tf.compat.v1.SparseTensorValue(indices = self.edges, values = attention_val, dense_shape = (self.n_samples, self.n_samples)) # dim: N × N
        attention_out = tf.sparse.softmax(attention_out)
        
        neighbor_out_info = tf.sparse.sparse_dense_matmul(attention_out, x_filtered)
        
        if FLAGS.directed == 1:
            #edges_receiver2sender = tf.concat([self.ind_receiver, self.ind_sender], axis = 1)
            edges_reverse = tf.concat([ind_receiver, ind_sender], axis = 1)
            attention_in = tf.compat.v1.SparseTensorValue(indices = edges_reverse, values = attention_val, dense_shape = (self.n_samples, self.n_samples)) # dim: N × N
            attention_in = tf.sparse.softmax(attention_in)
            
            neighbor_in_info = tf.sparse.sparse_dense_matmul(attention_in, x_filtered)
            # update convolutional hidden states
            output = self.act(self.vars['weight_self'] * x_transformed + self.vars['weight_neighbor_out'] * neighbor_out_info + self.vars['weight_neighbor_in'] * neighbor_in_info) #+ self.vars['bias_merge'])
        else:
            output = self.act(self.vars['weight_self'] * x_transformed + self.vars['weight_neighbor'] * neighbor_out_info) #+ self.vars['bias_merge'])
        
        return output

    def apply_regularizer(self, regularizer):
        return 0
        #return regularizer(self.vars['weights'])

    
class GraphConvolutionSparse1(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, outdegrees, indegrees, edges, num_nodes, features_nonzero, #placeholders, self_activation,
                 act = tf.nn.relu, dropout=0., **kwargs):
        #super(GraphConvolutionSparse, self).__init__(**kwargs)
        super().__init__(**kwargs)
        
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name = "weights")
            self.vars['bias'] = tf.Variable(tf.zeros((output_dim)), name = "bias")
            
        with tf.compat.v1.variable_scope(self.name + '_attention_vars'):
            self.vars['phi_attention'] = weight_variable_glorot(2 * output_dim, 1, name = 'weights_attention')
            self.vars['bias_attention'] = tf.Variable(tf.zeros((1)), name = "bias_attention")
        """
        self.gating_weight_dim = int(FLAGS.gating_weight)
        with tf.compat.v1.variable_scope(self.name + '_gate_vars'):
            self.vars['gating_weight1'] = weight_variable_glorot(input_dim, self.gating_weight_dim, name = 'gating_weight1')

            self.vars['gating_weight2'] = weight_variable_glorot(self.gating_weight_dim, output_dim, name = 'gating_weight2')
                                                
            self.vars['gating_bias1'] = weight_variable_glorot(1, self.gating_weight_dim, name = 'gating_bias1')

            self.vars['gating_bias2'] = tf.Variable(tf.zeros((1)), name = "gating_bias2")
        """
        if FLAGS.directed == 1:
            with tf.compat.v1.variable_scope(self.name + '_degree_vars'):
                self.vars['weight_degree_out'] = weight_variable_glorot(1, 1, name = "weight_degree_out")
                self.vars['weight_degree_in'] = weight_variable_glorot(1, 1, name = "weight_degree_in")
                #self.vars['bias_degree'] = tf.Variable(tf.zeros((1)), name = "bias_degree")
            
            with tf.compat.v1.variable_scope(self.name + '_merge_vars'):
                self.vars['weight_self'] = weight_variable_glorot(1, 1, name = 'weight_self')
                self.vars['weight_neighbor_out'] = weight_variable_glorot(1, 1, name = 'weight_neighbor_out')
                self.vars['weight_neighbor_in'] = weight_variable_glorot(1, 1, name = 'weight_neighbor_in')
                #self.vars['bias_merge'] = tf.Variable(tf.zeros((1)), name = "bias_merge")
        
        else:
            with tf.compat.v1.variable_scope(self.name + '_degree_vars'):
                self.vars['weight_degree'] = weight_variable_glorot(1, 1, name = "weight_degree")
                #self.vars['bias_degree'] = tf.Variable(tf.zeros((1)), name = "bias_degree")
            
            with tf.compat.v1.variable_scope(self.name + '_merge_vars'):
                self.vars['weight_self'] = weight_variable_glorot(1, 1, name = 'weight_self')
                self.vars['weight_neighbor'] = weight_variable_glorot(1, 1, name = 'weight_neighbor')
                #self.vars['bias_merge'] = tf.Variable(tf.zeros((1)), name = "bias_merge")
        
            
        #self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_samples = num_nodes
        self.act = act
        self.outdegrees = outdegrees
        self.indegrees = indegrees
        self.edges = edges
        #self.ind_sender = ind_sender
        #self.ind_receiver = ind_receiver
        self.dropout = dropout
        self.adj = adj
        self.issparse = True
        self.features_nonzero = features_nonzero
        #self.self_activation = self_activation # p_v
        #self.placeholders = placeholders
        #tf.compat.v1.set_random_seed(-1)
        # helper variable for sparse dropout

    def _call(self, inputs):
        x = inputs # N × K^l
        x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
        #feature transformation for hidden representation
        x_transformed = tf.sparse.sparse_dense_matmul(x, self.vars['weights']) + self.vars['bias'] # dim: N × K^(l + 1)
        x_transformed = tf.sparse.sparse_dense_matmul(self.adj, x_transformed)
        
        if FLAGS.directed == 1:
            gate_degree = tf.nn.sigmoid(self.vars['weight_degree_out'] * tf.reshape(self.outdegrees, [-1, 1]) + self.vars['weight_degree_in'] * tf.reshape(self.indegrees, [-1, 1])) #+ self.vars['bias_degree']) # dim: N × 1
        else:
            gate_degree = tf.nn.sigmoid(self.vars['weight_degree'] * tf.reshape(self.outdegrees, [-1, 1])) # dim: N × 1
        
        #gate = tf.reshape(tf.nn.elu(tf.sparse.sparse_dense_matmul(
        #                            tf.nn.elu(tf.sparse.sparse_dense_matmul(
        #                                tf.reshape(x, [-1, 1]), self.vars['gating_weight1' ]) + self.vars['gating_bias1' ]),
        #                            self.vars['gating_weight2' ]) + self.vars['gating_bias2' ]), [-1, 1])
        x_filtered = tf.multiply(x_transformed, gate_degree) # dim: N × K^(l + 1)
        #x_filtered = x_transformed # dim: N × K^(l + 1)

        #get self and neighbor representations for attention mechanism
        [ind_sender, ind_receiver] = tf.split(self.edges, num_or_size_splits = 2, axis = 1) # indices of senders (row) and recievers (column)
        #repre_neighbor = tf.nn.embedding_lookup(params = x_transformed, ids = tf.reduce_sum(self.ind_receiver, axis = 1)) # dim: edge_number × K^(l + 1)
        repre_neighbor = tf.nn.embedding_lookup(params = x_transformed, ids = tf.reduce_sum(ind_receiver, axis = 1)) # dim: edge_number × K^(l + 1)
        #repre_self = tf.nn.embedding_lookup(params = x_transformed, ids = tf.reduce_sum(self.ind_sender, axis = 1)) # dim: edge_num × K^(l + 1)
        repre_self = tf.nn.embedding_lookup(params = x_transformed, ids = tf.reduce_sum(ind_sender, axis = 1)) # dim: edge_num × K^(l + 1)
        repre_concat = tf.concat([repre_neighbor, repre_self], axis = 1) # dim: edge_num × 2*K^(l + 1)

        # calculate attention weight
        attention_val = tf.reshape(tf.nn.leaky_relu(tf.matmul(repre_concat, self.vars['phi_attention']) + self.vars['bias_attention'], alpha = 0.2), [-1]) # dim: edge_num
        #edges_sender2receiver = tf.concat([self.ind_sender, self.ind_receiver], axis = 1)
        #attention_out = tf.compat.v1.SparseTensorValue(indices = edges_sender2receiver, values = attention_val, dense_shape = (self.n_samples, self.n_samples)) # dim: N × N
        attention_out = tf.compat.v1.SparseTensorValue(indices = self.edges, values = attention_val, dense_shape = (self.n_samples, self.n_samples)) # dim: N × N
        attention_out = tf.sparse.softmax(attention_out)
        
        neighbor_out_info = tf.sparse.sparse_dense_matmul(attention_out, x_filtered)
        
        if FLAGS.directed == 1:
            #edges_receiver2sender = tf.concat([self.ind_receiver, self.ind_sender], axis = 1)
            edges_reverse = tf.concat([ind_receiver, ind_sender], axis = 1)
            attention_in = tf.compat.v1.SparseTensorValue(indices = edges_reverse, values = attention_val, dense_shape = (self.n_samples, self.n_samples)) # dim: N × N
            attention_in = tf.sparse.softmax(attention_in)
            
            neighbor_in_info = tf.sparse.sparse_dense_matmul(attention_in, x_filtered)
            # update convolutional hidden states
            output = self.act(self.vars['weight_self'] * x_transformed + self.vars['weight_neighbor_out'] * neighbor_out_info + self.vars['weight_neighbor_in'] * neighbor_in_info) #+ self.vars['bias_merge'])
        else:
            output = self.act(self.vars['weight_self'] * x_transformed + self.vars['weight_neighbor'] * neighbor_out_info) #+ self.vars['bias_merge'])

        return output

    def apply_regularizer(self, regularizer):
        return 0

class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, outdegrees, indegrees, edges, num_nodes, act = tf.nn.relu, dropout=0., **kwargs):
        #super(GraphConvolution, self).__init__(**kwargs)
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

    def apply_regularizer(self, regularizer):
        return 0
        #return regularizer(self.vars['weights'])

    
class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, outdegrees, indegrees, edges, num_nodes, features_nonzero, #placeholders, self_activation,
                 act = tf.nn.relu, dropout=0., **kwargs):
        #super(GraphConvolutionSparse, self).__init__(**kwargs)
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
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse.sparse_dense_matmul(x, self.vars['weights']) #+ self.vars['bias']
        x = tf.sparse.sparse_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

    def apply_regularizer(self, regularizer):
        return 0
        #return regularizer(self.vars['weights'])

class WeightedInnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(WeightedInnerProductDecoder, self).__init__(**kwargs)
        with tf.compat.v1.variable_scope(self.name + '_weight'):
            self.vars['weights'] = matrix_weight_variable_truncated_normal(input_dim, name="matrix_weight")
        self.dropout = dropout
        self.act = act

    def get_weight_matrix(self):
        W = (self.vars['weights'] + tf.transpose(a=self.vars['weights'])) * 1/2
        return W
    
    def _call(self, inputs):

        W = (self.vars['weights'] + tf.transpose(a=self.vars['weights'])) * 1/2
        
        inputs = tf.nn.dropout(inputs, self.dropout)
        x = tf.transpose(a=inputs)
        #inputs = inputs + tf.matmul(inputs, W)
        inputs = tf.matmul(inputs, W)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs

    def apply_regularizer(self, regularizer):
        return regularizer(self.vars['weights'])

class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., normalize = False, act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.input_dim = input_dim
        self.normalize = normalize

    def _call(self, inputs):
        if(self.normalize):
            inputs = tf.nn.l2_normalize(inputs, axis=1)
        
        inputs = tf.nn.dropout(inputs, self.dropout)
        x = tf.transpose(a=inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs
    
    def get_weight_matrix(self):
        W = tf.eye(self.input_dim)
        return W

    def apply_regularizer(self, regularizer):
        return tf.constant(0.0)
    
class LSMDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, num_nodes, act = tf.nn.sigmoid, dropout = 0., **kwargs):
        #super(LSMDecoder, self).__init__(**kwargs)
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
        #else:
            #with tf.compat.v1.variable_scope(self.name + '_vars_weights'):
                #self.vars['weight_gam'] = tf.Variable(tf.ones(1), name = "weight_gam")

    def _call(self, inputs):
        #inputs = tf.nn.dropout(inputs, self.dropout)
        z = inputs[0]
        z_decoder = tf.matmul(z, self.vars['transform_z'])
        gamma = inputs[3]
        gamma_decoder = tf.matmul(gamma, self.vars['transform_gam'])
        #z_decoder = []
        #for d in range(self.latent_dim):
        #    z_decoder.append(tf.matmul(z[d], self.vars['weights_z_' + str(d)]))
        #z_decoder = tf.transpose(tf.reshape(z_decoder, [self.latent_dim, -1])) # dim: N × d
        dist_gam = self.n_samples * distance_scaled_by_gam(z_decoder, gamma_decoder + SMALL)
        #dist_gam = distance_scaled_by_gam(z, gamma + SMALL)
        #dist_gam = tf.compat.v1.check_numerics(dist_gam, 'dist_gam is nan')
        
        if FLAGS.directed == 1:
            delta = inputs[4]
            delta_decoder = tf.matmul(delta, self.vars['transform_del'])
            dist_del = self.n_samples * distance_scaled_by_del(z_decoder, delta_decoder + SMALL)
            #dist_del = tf.compat.v1.check_numerics(dist_del, 'dist_del is nan')
            #delta_decoder = tf.compat.v1.check_numerics(delta_decoder, 'delta_decoder is nan')
            
            x = self.vars['bias'] - self.vars['weight_gam'] * dist_gam - self.vars['weight_del'] * dist_del
        else:
            delta_decoder = gamma_decoder
            x = self.vars['bias'] - self.vars['weight_gam'] * (dist_gam + tf.transpose(dist_gam))
            #x = self.vars['bias'] - tf.multiply(dist, self.vars['weight_gam'] / gamma_decoder + self.vars['weight_gam'] / delta_decoder)

        #x = self.vars['bias'] - dist + gamma_decoder + delta_decoder
        x = tf.reshape(x, [-1])
        #x = tf.compat.v1.check_numerics(x, 'logit is nan')
        output = self.act(x)
        #output = tf.compat.v1.check_numerics(output, 'reconstruction is nan')
        #return output, (z_decoder, gamma_decoder, delta_decoder, self.vars['weight_gam'], self.vars['weight_del'])
        return output, (z_decoder, gamma_decoder, delta_decoder)
    
    def get_weight_matrix(self):
        W = tf.eye(self.input_dim)
        return W
    
    def apply_regularizer(self, regularizer):
        return tf.constant(0.0)

class WeightedInnerProductDecoder2(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(PosInnerProductDecoder, self).__init__(**kwargs)
        with tf.compat.v1.variable_scope(self.name + '_weight'):
            self.vars['weights'] = matrix_weight_variable_normal(input_dim, scale=FLAGS.lambda_mat_scale, name="matrix_weight")

        self.dropout = dropout
        self.act = act

    def get_weight_matrix(self):
        W = self.vars['weights']
        W = (W + tf.transpose(a=W)) * 1/2
        #W = tf.nn.sigmoid(W)
        #W = tf.nn.softmax(W);
        return W
    
    def _call(self, inputs):

        W = self.get_weight_matrix()
        inputs = tf.nn.dropout(inputs, self.dropout)
        x = tf.transpose(a=inputs)
        #inputs = inputs + tf.matmul(inputs, W)
        inputs = tf.matmul(inputs, W)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
	
        return outputs

    def apply_regularizer(self, regularizer):
        return regularizer(self.vars['weights'])

class PosInnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(PosInnerProductDecoder, self).__init__(**kwargs)
        with tf.compat.v1.variable_scope(self.name + '_weight'):
            self.vars['weights'] = matrix_weight_variable_normal(input_dim, scale=FLAGS.lambda_mat_scale, name="matrix_weight")

        self.dropout = dropout
        self.act = act

    def get_weight_matrix(self):
        W = self.vars['weights']
        W = (W + tf.transpose(a=W)) * 1/2
        W = tf.nn.sigmoid(W)
        #W = tf.nn.softmax(W);
        return W
    
    def _call(self, inputs):

        W = self.get_weight_matrix()
        inputs = tf.nn.dropout(inputs, self.dropout)
        x = tf.transpose(a=inputs)
        #inputs = inputs + tf.matmul(inputs, W)
        inputs = tf.matmul(inputs, W)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
	
        return outputs

    def apply_regularizer(self, regularizer):
        return regularizer(self.vars['weights'])

class DiagonalInnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., normalize = False, act=tf.nn.sigmoid, **kwargs):
        super(DiagonalInnerProductDecoder, self).__init__(**kwargs)
        with tf.compat.v1.variable_scope(self.name + '_weight'):
            self.vars['weights'] = vector_weight_variable_truncated_normal((1, input_dim), name="matrix_weight", scale=0.1)

        self.dropout = dropout
        self.act = act
        self.normalize = normalize
    
    def _call(self, inputs):

        if(self.normalize):
            inputs = tf.nn.l2_normalize(inputs, axis=1)
        
        W = self.get_weight_matrix()#self.vars['weights'];#tf.nn.sigmoid(self.vars['weights'])
        inputs = tf.nn.dropout(inputs, self.dropout)
        x = tf.transpose(a=inputs)
        inputs = inputs * W
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
	
        return outputs

    def apply_regularizer(self, regularizer):
        return regularizer(self.vars['weights'])

    
    def get_weight_matrix(self):
        W = self.vars['weights']
        #W = (W + tf.transpose(W)) * 1/2
        W = tf.nn.sigmoid(W)
        #W = tf.nn.softmax(W);
        return W

class batch_norm(object):
     
     #def __init__(self, epsilon=1e-5, momentum = 0.99, name="batch_norm"):
     def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
          with tf.compat.v1.variable_scope(name):
               self.epsilon  = epsilon
               self.momentum = momentum
               self.name = name

     def __call__(self, x, phase):
          return tf.contrib.layers.batch_norm(x,
                                              decay=self.momentum, 
                                              epsilon=self.epsilon,
                                              scale=True,
                                              center=True, 
                                              is_training=phase,
                                              scope=self.name)
