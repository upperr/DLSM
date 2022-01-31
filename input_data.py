import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    
    if dataset.startswith('wiki'):
        adj, features, feature_presence, directed = load_wiki_mat(dataset)
        return adj, features, feature_presence, directed
    
    elif dataset.startswith('political'):
        adj, features, feature_presence, directed = load_political_mat(dataset)
        return adj, features, feature_presence, directed
    
    elif dataset.startswith('email'):
        adj, features, feature_presence, directed = load_email_mat(dataset)
        return adj, features, feature_presence, directed
    
    elif dataset.startswith('google'):
        adj, features, feature_presence, directed = load_google_mat(dataset)
        return adj, features, feature_presence, directed
    else:
        raise NameError('No data named ' + dataset + '.')

def load_political_mat(dataset):

    mat_data = sio.loadmat('data/political.mat')

    adj = mat_data['adj']
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))

    return adj, features, 0, True

def load_wiki_mat(dataset):

    mat_data = sio.loadmat('data/Wiki.mat')

    adj = mat_data['adj']
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))

    return adj, features, 0, True

def load_email_mat(dataset):

    mat_data = sio.loadmat('data/email.mat')

    adj = mat_data['adj']
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))

    return adj, features, 0, True

def load_google_mat(dataset):

    adj = nx.adjacency_matrix(nx.read_edgelist("data/GoogleNw.txt",
                                                   delimiter='\t',
                                                   create_using=nx.DiGraph()))
    features = sp.identity((adj.shape[0]))

    return adj, features, 0, False
    
def load_training_data(dataset_str, split_idx=0):

    data_path = 'data/' + dataset_str + '/split_' + str(split_idx) + '.npz'
    data = np.load(data_path, allow_pickle = True, encoding = 'latin1')
    
    return data['adj_train'], data['val_edges'], data['val_edges_false'], data['test_edges'], data['test_edges_false']

def load_node_labels(dataset):
    
    if dataset.startswith('email'):
        label = np.genfromtxt('data/email_comm.txt', delimiter = ' ', dtype = 'int32')
        
    elif dataset.startswith('political'):
        comm = pd.read_csv('data/political_comm.csv')
        label = np.zeros(len(comm), dtype = 'int32')
        label[comm['comm'] == 'con'] = 1
        
    else:
        raise NameError('Dataset ' + dataset + ' has no node labels.')
        
    return label