from __future__ import division
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import networkx as nx
import powerlaw

SMALL = 1e-16
SMALL2 = 1e-8
EULER_GAMMA = 0.5772156649015329


def logit(x):
    return tf.math.log(x + SMALL2) - tf.math.log(1. - x + SMALL2)

def log_density_logistic(logalpha, sample, temperature):
    """
    log-density of the Logistic distribution, from 
    Maddison et. al. (2017) (right after equation 26)
    Input logalpha is a logit (alpha is a probability ratio)
    """
    exp_term = logalpha - sample * temperature
    log_prob = exp_term + np.log(temperature) - 2. * tf.nn.softplus(exp_term)
    return log_prob

def sample_normal(mean, log_std):

    # mu + standard_samples * stand_deviation
    x = mean + tf.random.normal(tf.shape(mean)) * tf.exp(log_std)
    return x

def sample_bernoulli(p_logit):

    p = tf.nn.sigmoid(p_logit)
    u = tf.random.uniform(tf.shape(p_logit), 1e-4, 1. - 1e-4)
    x = tf.round(u > p)
        
    return x

def sample_binconcrete(pi_logit, temperature):

    # Concrete instead of Bernoulli
    u = tf.random.uniform(tf.shape(pi_logit), 1e-4, 1. - 1e-4)
    L = tf.math.log(u) - tf.math.log(1. - u)
    x_logit = (pi_logit + L) / temperature
    #s = tf.sigmoid(logit)
        
    return x_logit

def sample_gamma(alpha, beta):

    u = tf.random.uniform(tf.shape(alpha), 1e-4, 1. - 1e-4)
    x = tf.exp(-tf.math.log(beta + SMALL) + (tf.math.log(u) + tf.math.log(alpha + SMALL) + tf.math.lgamma(alpha + SMALL)) / (alpha + SMALL))
        
    return x

def kl_normal(mean_posterior, log_std, mean_prior = 0.):
    #mean, log_std: d × N × K
    kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + 2 * log_std - tf.square(mean_posterior - mean_prior) - tf.square(tf.exp(log_std)), axis = 1))
    return kl

def kl_binconcrete(logit_posterior, logit_prior, sample, temp_posterior, temp_prior):
    """
    KL divergence between the prior and posterior
    inputs are in logit-space
    """
    log_prior = log_density_logistic(logit_prior, sample, temp_prior)
    log_posterior = log_density_logistic(logit_posterior, sample, temp_posterior)
    kl = log_posterior - log_prior
    return tf.reduce_mean(tf.reduce_sum(kl, axis=1))

def kl_gamma(alpha_posterior, alpha_prior):
    """
    KL divergence between the prior and posterior
    """
    kl = tf.math.lgamma(alpha_prior) - tf.math.lgamma(alpha_posterior + SMALL) + (alpha_posterior - alpha_prior) * tf.math.digamma(alpha_posterior + SMALL)
    return tf.reduce_mean(tf.reduce_sum(kl, axis = 1))
    
def Euclidean_dist(x):

    squared_sum = tf.reduce_sum(tf.square(x), axis=1)
    squared_sum = tf.reshape(squared_sum, [-1, 1])  # Column vector.
  
    squared_dis = squared_sum - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(squared_sum)
    euclidean_dis = tf.sqrt(tf.add(squared_dis, SMALL2))
    
    return euclidean_dis

def distance_scaled_by_gam(x, gamma):

    gamma_squared = tf.square(gamma)
    x_scaled = x * gamma_squared
    x_squared_scaled = tf.matmul(gamma_squared, tf.transpose(tf.square(x)))
    x_squared_scaled_diag = tf.reshape(tf.compat.v1.diag_part(x_squared_scaled), [-1, 1]) # column vector

    distance_squared = x_squared_scaled + x_squared_scaled_diag - 2 * tf.matmul(x_scaled, tf.transpose(x))
    distance = tf.sqrt(distance_squared + SMALL2)

    return distance

def distance_scaled_by_del(x, delta):

    delta_squared = tf.square(delta)
    x_scaled = x * delta_squared
    x_squared_scaled = tf.matmul(tf.square(x), tf.transpose(delta_squared))
    x_squared_scaled_diag = tf.reshape(tf.compat.v1.diag_part(x_squared_scaled), [1, -1]) # row vector
  
    distance_squared = x_squared_scaled + x_squared_scaled_diag - 2 * tf.matmul(x, tf.transpose(x_scaled))
    distance = tf.sqrt(distance_squared + SMALL2)

    return distance

def statistics_degrees(A):
    """
    Compute statistics of degrees

    Parameters
    ----------
    A: sparse matrix or np.array. The input adjacency matrix.
    Returns
    -------
    max out-degree, max in-degree, mean degree
    """

    degrees_in = A.sum(axis = 0)
    degrees_out = A.sum(axis = 1)
    return np.max(degrees_out), np.max(degrees_in), np.mean(degrees_out)

def statistics_transitivity_rate(A):
    """
    Compute the transitivity rate of the input graph

    Parameters
    ----------
    A: sparse matrix or np.array. The input adjacency matrix.
    
    Returns
    -------
    Transitive link rate
    """
    
    A_sym = A + A.T # transform an asymmetric adjacency matrix to undirected
    A_sym2 = A_sym.dot(A_sym)
    if isinstance(A_sym2, np.ndarray):
        A_sym2 = A_sym2 - np.diag(A_sym2.diagonal())
        two_hops_num = A_sym2.sum() * 2 # total number of two-hop paths between arbitrary ordered node pairs
        transitive_num = (A_sym * A_sym2).sum() # number of transitive links
    else:
        A_sym2 = A_sym2 - sp.dia_matrix((A_sym2.diagonal()[np.newaxis, :], [0]), shape = A_sym2.shape)
        two_hops_num = A_sym2.sum() * 2 # total number of two-hop paths between arbitrary ordered node pairs
        transitive_num = A_sym.multiply(A_sym2).sum() # number of transitive links
    
    return transitive_num / two_hops_num

def statistics_reciprocity_rate(A):
    """
    Compute the reciprocity rate of the input graph

    Parameters
    ----------
    A: sparse matrix or np.array. The input adjacency matrix.
    
    Returns
    -------
    reciprocal link rate
    """
    
    if isinstance(A, np.ndarray):
        reciprocal_num = (A * A.T).sum()
    else:
        reciprocal_num = A.multiply(A.T).sum()
    
    return reciprocal_num / A.sum()

def statistics_power_law_alpha(A):
    """
    Compute the power law coefficient of the degree distribution of the input graph

    Parameters
    ----------
    A: sparse matrix or np.array. The input adjacency matrix.

    Returns
    -------
    Power law coefficient
    """
    
    degrees_in = np.reshape(np.array(A.sum(axis = 0)), [-1])
    degrees_out = np.reshape(np.array(A.sum(axis = 1)), [-1])
    powerlaw_out = powerlaw.Fit(degrees_out, xmin = max(np.min(degrees_out), 1)).power_law.alpha
    powerlaw_in = powerlaw.Fit(degrees_in, xmin = max(np.min(degrees_in), 1)).power_law.alpha
    return powerlaw_out, powerlaw_in

def statistics_claw_count(A):
    """
    Compute the claw count of the input graph

    Parameters
    ----------
    A: sparse matrix or np.array. The input adjacency matrix.

    Returns
    -------
    Claw count
    """
    A = A + A.T
    A[A > 1] = 1
    degrees = np.reshape(np.array(A.sum(axis=0)), [-1])

    #degrees_total = np.reshape(np.array(A.sum(0) + A.sum(1)), [-1])
    return float(np.sum(np.array([1 / 6. * x * (x - 1) * (x - 2) for x in degrees])))

def statistics_triangle_count(A):
    """
    Compute the triangle count of the input graph

    Parameters
    ----------
    A: sparse matrix or np.array. The input adjacency matrix.
    
    Returns
    -------
    Triangle count
    """
    
    if isinstance(A, np.ndarray):
        A_graph = nx.from_numpy_matrix(A)
    else:
        A_graph = nx.from_numpy_matrix(A.toarray())
    triangles = nx.triangles(A_graph)
    t = np.sum(list(triangles.values())) / 3
    return int(t)

def clustering_coefficient(A):
    """
    Compute the clustering coefficient of the input graph
    See Fagiolo (2007) for more details.

    Parameters
    ----------
    A: sparse matrix or np.array. The input adjacency matrix.

    Returns
    -------
    Clustering coefficient
    """
    
    degree_out = np.reshape(np.array(A.sum(1)), [-1])
    degree_in = np.reshape(np.array(A.sum(0)), [-1])
    degree_total = degree_out + degree_in
    degree_bilateral = A.dot(A).diagonal()
    
    A_sym = A + A.T
    num_triangles = A_sym.dot(A_sym).dot(A_sym).diagonal() / 2
    num_triangles_pos = degree_total * (degree_total - 1) - 2 * degree_bilateral
    CC = num_triangles[num_triangles_pos > 0] / num_triangles_pos[num_triangles_pos > 0]
    CC_avg = np.mean(CC)
    
    return CC_avg

def compute_graph_statistics(A):
    """
    Parameters
    ----------
    A_in: sparse matrix
          The input adjacency matrix.
          
    Returns
    -------
    Dictionary containing the following statistics:
             * Maximum, minimum, mean degree of nodes
             * Power law exponent
             * Clustering coefficient
             * Number of connected components
    """

    statistics = {}
    d_out_max, d_in_max, d_mean = statistics_degrees(A)
    # Degree statistics
    statistics['d_out_max'] = d_out_max
    statistics['d_in_max'] = d_in_max
    statistics['d_mean'] = d_mean
    # transitivity rate
    statistics['transitivity_rate'] = statistics_transitivity_rate(A)
    # reciprocity rate
    statistics['reciprocity_rate'] = statistics_reciprocity_rate(A)
    # power law exponent
    statistics['power_law_exp_out'], statistics['power_law_exp_in'] = statistics_power_law_alpha(A)
    # Clustering coefficient
    #statistics['clustering_coefficient'] = 3 * statistics_triangle_count(A) / statistics_claw_count(A)
    statistics['clustering_coefficient'] = clustering_coefficient(A)

    return statistics