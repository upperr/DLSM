import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SMALL = 1e-16

def parse_args():
    '''
    Parses the arguments for visualization.
    '''
    parser = argparse.ArgumentParser(description = "Visualization")
    parser.add_argument('--file_emb', nargs = '?', help = 'The .npz file of generated node embeddings')
    parser.add_argument('--dataset', nargs = '?', help = 'Dataset to visulize: email, political')
    parser.add_argument('--dim', type = int, default = 2, help = 'Dimension of latent positions for visualization')
    return parser.parse_args()

def cal_pairwise_dist(x):
    '''Compute pairwise distances, x is a matrix
    (a-b)^2 = a^w + b^2 - 2*a*b
    '''
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    return dist
 
def cal_perplexity(dist, idx=0, beta=1.0):
    '''Compute perplexity (here only the entropy is computed for convenience)
    dist is a distance vector;
    idx is the index in the distance vector;
    beta is the sigma of Gaussian distribution.
    '''
    prob = np.exp(-dist * beta)
    # set the own prob to be 0
    prob[idx] = 0
    sum_prob = np.sum(prob)
    perp = np.log(sum_prob + SMALL) + beta * np.sum(dist * prob) / (sum_prob + SMALL)
    prob /= sum_prob
    return perp, prob
 
def seach_prob(x, tol=1e-5, perplexity=30.0):
    '''Search beta using dichotomy and compute pairwise prob'''
    # Initialization
    print("Computing pairwise distances...")
    (n, d) = x.shape
    dist = cal_pairwise_dist(x)
    pair_prob = np.zeros((n, n))
    beta = np.ones((n, 1))
    # use log for the simplicity of computations
    base_perp = np.log(perplexity)
 
    for i in range(n):
        if i % 500 == 0:
            print("Computing pair_prob for point %s of %s ..." %(i,n))
 
        betamin = -np.inf
        betamax = np.inf
        perp, this_prob = cal_perplexity(dist[i], i, beta[i])
 
        # search the prob with best sigma via dichotomy
        perp_diff = perp - base_perp
        tries = 0
        while np.abs(perp_diff) > tol and tries < 50:
            if perp_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2
                else:
                    beta[i] = (beta[i] + betamin) / 2
 
            # update perb, prob
            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1
        # record prob
        pair_prob[i,] = this_prob
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    return pair_prob
 
def pca(x, no_dims = 50):
    ''' Use PCA to reduce dimensions
    '''
    print("Preprocessing the data using PCA...")
    (n, d) = x.shape
    x = x - np.tile(np.mean(x, 0), (n, 1))
    l, M = np.linalg.eig(np.dot(x.T, x))
    y = np.dot(x, M[:,0:no_dims])
    return y
 
def tsne(x, no_dims=2, initial_dims=50, perplexity=30.0, max_iter=1000):
    """Runs t-SNE on the dataset in the NxD array x
    to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(x, no_dims, perplexity),
    where x is an NxD NumPy array.
    """
 
    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array x should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1
 
    # Initialization
    x = pca(x, initial_dims).real
    (n, d) = x.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    y = np.random.randn(n, no_dims)
    dy = np.zeros((n, no_dims))
    iy = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
 
    # Symmetrization
    P = seach_prob(x, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    # early exaggeration
    P = P * 4
    P = np.maximum(P, 1e-12)
 
    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_y = np.sum(np.square(y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
 
        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dy[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (y[i,:] - y), 0)
 
        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dy > 0) != (iy > 0)) + (gains * 0.8) * ((dy > 0) == (iy > 0))
        gains[gains < min_gain] = min_gain
        iy = momentum * iy - eta * (gains * dy)
        y = y + iy
        y = y - np.tile(np.mean(y, 0), (n, 1))
        # Compute current value of cost function
        if (iter + 1) % 100 == 0:
            if iter > 100:
                C = np.sum(P * np.log(P / Q))
            else:
                C = np.sum( P/4 * np.log( P/4 / Q))
            print("Iteration", (iter + 1), ": error is", C)
        # Stop lying about P-values
        if iter == 100:
            P = P / 4
    print("finished training!")
    return y

if __name__ == "__main__":
    args = parse_args()
    
    est = np.load('data/' + args.file_emb, allow_pickle = True)
    if args.dataset == 'email':
        label = np.genfromtxt('data/email_comm.txt', delimiter = ' ', dtype = 'int32')
    elif args.dataset == 'political':
        label_raw = pd.read_csv('data/political_comm.csv')
        label = np.zeros(len(label_raw), dtype = 'int32')
        label[label_raw['comm'] == 'con'] = 1
    else:
        raise NameError('No node labels for dataset ' + args.dataset + '!')
    
    if not os.path.exists('fig/'):
        os.makedirs('fig/')

    theta = est['theta_decoder']
    z = theta[0]
    ## latent position visualization ##
    coor = tsne(z, args.dim, z.shape[1], 20.0)

    plt.figure(figsize = (5, 4))
    plt.scatter(coor[:, 0], coor[:, 1], s = 3, c = label, cmap = 'rainbow')
    plt.xlabel(r'$\tilde{z}_1$')
    plt.ylabel(r'$\tilde{z}_2$')
    plt.savefig('fig/' + args.dataset + '_tsne.pdf')
    plt.show()
