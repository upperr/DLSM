import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset):
    
    if dataset.startswith('nips12'):
        adj, features, feature_presence, directed = load_nips_mat(dataset)
        return adj, features, feature_presence, directed
        
    elif dataset.startswith('kohonen'):
        adj, features, feature_presence, directed = load_kohonen_mat(dataset)
        return adj, features, feature_presence, directed
    
    elif dataset.startswith('wiki'):
        adj, features, feature_presence, directed = load_wiki_mat(dataset)
        return adj, features, feature_presence, directed
    
    elif dataset.startswith('political'):
        adj, features, feature_presence, directed = load_political_mat(dataset)
        return adj, features, feature_presence, directed
    
    elif dataset.startswith('ciaodvd'):
        adj, features, feature_presence, directed = load_ciaodvd_mat(dataset)
        return adj, features, feature_presence, directed
    
    elif dataset.startswith('dblp'):
        adj, features, feature_presence, directed = load_dblp_mat(dataset)
        return adj, features, feature_presence, directed
    
    elif dataset.startswith('email'):
        adj, features, feature_presence, directed = load_email_mat(dataset)
        return adj, features, feature_presence, directed
    
    elif dataset.startswith('british'):
        adj, features, feature_presence, directed = load_british_mat(dataset)
        return adj, features, feature_presence, directed
    
    elif dataset.startswith('crocodile'):
        adj, features, feature_presence, directed = load_crocodile_mat(dataset)
        return adj, features, feature_presence, directed
    
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding = 'latin1'))
    x, tx, allx, graph = tuple(objects)

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil() # convert to linked list
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features, 1, False

def load_nips_mat(dataset):

    mat_data = sio.loadmat('data/nips12.mat')

    adj = mat_data['B']
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))

    return adj, features, 0, False

def load_kohonen_mat(dataset):

    mat_data = sio.loadmat('data/Kohonen.mat')

    adj = mat_data['adj']
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))

    return adj, features, 0, True

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

def load_ciaodvd_mat(dataset):

    mat_data = sio.loadmat('data/CiaoDVD.mat')

    adj = mat_data['adj']
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))

    return adj, features, 0, True

def load_dblp_mat(dataset):

    mat_data = sio.loadmat('data/DBLP.mat')

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

def load_british_mat(dataset):

    mat_data = sio.loadmat('data/british.mat')

    adj = mat_data['adj']
    adj = sp.csr_matrix(adj)
    features = sp.identity((adj.shape[0]))

    return adj, features, 0, False

def load_crocodile_mat(dataset):

    mat_data = sio.loadmat('data/crocodile.mat')

    adj = mat_data['adj']
    adj = sp.csr_matrix(adj)
    features = mat_data['features']
    features = sp.lil_matrix(features)

    return adj, features, 1, True
    
def load_masked_test_edges(dataset_str, split_idx=0):

    data_path = 'data/' + dataset_str + '/split_' + str(split_idx) + '.npz'
    data = np.load(data_path, allow_pickle = True, encoding = 'latin1')
    
    return data['adj_train'], data['train_edges'], data['val_edges'], data['val_edges_false'], data['test_edges'], data['test_edges_false']
