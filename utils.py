from __future__ import division
import tensorflow as tf
import numpy as np

SMALL = 1e-16
SMALL2 = 1e-8
SMALL3 = 1e-2
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

def Beta_fn(a, b):
    beta_ab = tf.exp(tf.math.lgamma(a) + tf.math.lgamma(b) - tf.math.lgamma(a + b))
    return beta_ab

def log_pdf_bernoulli(x,p):
    return x * tf.math.log(p + SMALL) + (1-x) * tf.math.log(1-p + SMALL)

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

def sample_kumaraswamy(a, b):
    
    u = tf.random.uniform(tf.shape(a), 1e-4, 1. - 1e-4)
    # return (1. - u.pow(1./b)).pow(1./a)
    x = tf.exp(tf.math.log(1. - tf.exp(tf.math.log(u) / (b + SMALL)) + SMALL) / (a + SMALL))
    return x

def draw_weibull(k, l):
    # x ~ Weibull(k, l)
    #print k.shape
    uniform = tf.random.uniform(tf.shape(input=k), 1e-4, 1. - 1e-4)
    x = l * tf.pow(-tf.math.log(1-uniform), 1/k)
    return x

def kl_weibull_gamma(k_w, l_w, alpha_g, beta_g):
    # KL(Weibull(k_w, l_w)||Gamma(alpha_g, beta_g))
    # See eqn from paper.

    #print k_w.get_shape()
    #print l_w.get_shape()
    #print alpha_g.shape
    #sys.exit()
    
    #typecasting as log needed one of the floats
    k_w = tf.cast(k_w, tf.float32)
    l_w = tf.cast(l_w, tf.float32)
    alpha_g = tf.cast(alpha_g, tf.float32)
    beta_g = tf.cast(beta_g, tf.float32)

    kl = -alpha_g * tf.math.log(l_w + SMALL2) + (EULER_GAMMA * alpha_g) / (k_w+SMALL2) + tf.math.log(k_w + SMALL2) + beta_g * l_w * tf.exp(tf.math.lgamma(1 + (1 / (k_w+SMALL2) ))) - \
    EULER_GAMMA - 1 - alpha_g * tf.math.log(beta_g + SMALL2) + tf.math.lgamma(alpha_g+SMALL2)

    kl = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=kl, axis=1))
    #kl = tf.minimum(10e8, kl)
    #kl = tf.clip_by_value(kl, 0.0, 100.0)
   
    #Desperate times, desperate measures
    kl = tf.clip_by_value(kl, 0.0, 10e5)

    return kl

def kl_normal(mean_posterior, log_std, mean_prior = 0.):
    #mean, log_std: d × N × K
    kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + 2 * log_std - tf.square(mean_posterior - mean_prior) - tf.square(tf.exp(log_std)), axis = 1))
    return kl

def kl_bernoulli(p_posterior, p_prior):
    """
    KL divergence between the prior and posterior
    """
    addterm1 = p_posterior * (tf.math.log(p_posterior) - tf.math.log(p_prior))
    addterm2 = (1. - p_posterior) * (tf.math.log(1. - p_posterior) - tf.math.log(1. - p_prior))
    kl = addterm1 - addterm2
    return tf.reduce_mean(tf.reduce_sum(kl, axis = 1))

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

# the prior is default Beta(alpha_0, 1)
def kl_kumar_beta(a, b, prior_alpha = 10., log_beta_prior = np.log(1./10.)):
    """
    KL divergence between Kumaraswamy(a, b) and Beta(prior_alpha, prior_beta)
    as in Nalisnick & Smyth (2017) (12)
    - we require you to calculate the log of beta function, since that's a fixed quantity
    """
    prior_beta = 1.
    
    # digamma = b.log() - 1/(2. * b) - 1./(12 * b.pow(2)) # this doesn't seem to work
    first_term = ((a - prior_alpha)/(a+SMALL)) * (-1 * EULER_GAMMA - tf.math.digamma(b) - 1./(b+SMALL))
    second_term = tf.math.log(a+SMALL) + tf.math.log(b+SMALL) + log_beta_prior
    third_term = -(b - 1)/(b+SMALL)
    
    ab  = a*b + SMALL
    kl  = 1./(1+ab) * Beta_fn(1./(a+SMALL), b)
    kl += 1./(2+ab) * Beta_fn(2./(a+SMALL), b)
    kl += 1./(3+ab) * Beta_fn(3./(a+SMALL), b)
    kl += 1./(4+ab) * Beta_fn(4./(a+SMALL), b)
    kl += 1./(5+ab) * Beta_fn(5./(a+SMALL), b)
    kl += 1./(6+ab) * Beta_fn(6./(a+SMALL), b)
    kl += 1./(7+ab) * Beta_fn(7./(a+SMALL), b)
    kl += 1./(8+ab) * Beta_fn(8./(a+SMALL), b)
    kl += 1./(9+ab) * Beta_fn(9./(a+SMALL), b)
    kl += 1./(10+ab) * Beta_fn(10./(a+SMALL), b)
    kl *= (prior_beta-1)*b

    kl += first_term + second_term + third_term
    return tf.reduce_mean(tf.reduce_sum(kl, axis = 1))

def test_kl():

    from math import gamma
    
    k_w = 2
    l_w = 2

    alpha_g = 2
    beta_g = 2

    kl = -alpha_g * np.log(l_w) + (EULER_GAMMA * alpha_g) / k_w + np.log(k_w) + beta_g * l_w * gamma(1 + (1/k_w)) - \
         EULER_GAMMA - 1 - alpha_g * np.log(beta_g) + np.log(gamma(alpha_g))

    print(kl)
    #sys.exit()
    
def Euclidean_dist(x):

    squared_sum = tf.reduce_sum(tf.square(x), axis=1)
    squared_sum = tf.reshape(squared_sum, [-1, 1])  # Column vector.
  
    squared_dis = squared_sum - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(squared_sum)
    euclidean_dis = tf.sqrt(tf.add(squared_dis, SMALL3))
    #euclidean_dis = tf.clip_by_value(squared_dis, 1e-6, 10.0)
    return euclidean_dis

def distance_scaled_by_gam(x, gamma):

    gamma_squared = tf.square(gamma)
    #x_scaled = x / gamma_squared
    x_scaled = x * gamma_squared
    #x_squared_scaled = tf.matmul(1. / gamma_squared, tf.transpose(tf.square(x)))
    x_squared_scaled = tf.matmul(gamma_squared, tf.transpose(tf.square(x)))
    x_squared_scaled_diag = tf.reshape(tf.compat.v1.diag_part(x_squared_scaled), [-1, 1]) # column vector

    distance_squared = x_squared_scaled + x_squared_scaled_diag - 2 * tf.matmul(x_scaled, tf.transpose(x))
    #mindist = tf.sort(tf.reshape(distance_squared,[-1]))
    distance = tf.sqrt(distance_squared + SMALL2)
    #distance = tf.compat.v1.check_numerics(distance, 'distance_gam is nan')

    return distance#, mindist

def distance_scaled_by_del(x, delta):

    delta_squared = tf.square(delta)
    #x_scaled = x / delta_squared
    x_scaled = x * delta_squared
    #x_squared_scaled = tf.matmul(tf.square(x), tf.transpose(1. / delta_squared))
    x_squared_scaled = tf.matmul(tf.square(x), tf.transpose(delta_squared))
    x_squared_scaled_diag = tf.reshape(tf.compat.v1.diag_part(x_squared_scaled), [1, -1]) # row vector
  
    distance_squared = x_squared_scaled + x_squared_scaled_diag - 2 * tf.matmul(x, tf.transpose(x_scaled))
    #mindist = tf.sort(tf.reshape(distance_squared,[-1]))
    distance = tf.sqrt(distance_squared + SMALL2)
    #distance = tf.compat.v1.check_numerics(distance, 'distance_del is nan')

    return distance

def mds(dist, n_dims = 2):
    #compute the multidimensional scaling
    
    n = dist.shape[0]

    c = np.ones((n, n)) * np.sum(dist) / n ** 2
    c_row = np.sum(dist, axis = 1, keepdims = True) / n
    c_col = np.sum(dist, axis = 0, keepdims = True) / n

    B = -(c - c_row - c_col + dist) / 2

    eig_val, eig_vector = np.linalg.eig(B)
    index_ = np.argsort(-eig_val)[:n_dims]
    picked_eig_val = eig_val[index_].real
    picked_eig_vector = eig_vector[:, index_]
    # print(picked_eig_vector.shape, picked_eig_val.shape)
    return (picked_eig_vector * picked_eig_val ** (0.5)).real
