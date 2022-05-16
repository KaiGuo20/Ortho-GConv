import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import sys
from normalization import fetch_normalization, row_normalize
import pickle as pkl
import networkx as nx
import json
import sys
import os
from networkx.readwrite import json_graph

import pdb


sys.setrecursionlimit(99999)


def _preprocess_adj( normalization, adj, cuda):
    adj_normalizer = fetch_normalization(normalization)
    r_adj = adj_normalizer(adj)
    r_adj = sparse_mx_to_torch_sparse_tensor(r_adj).float()
    if cuda:
        r_adj = r_adj.cuda()
    return r_adj


def _preprocess_fea(fea, cuda):
    if cuda:
        return fea.cuda()
    else:
        return fea
def stub_sampler(train_adj,train_features,normalization, cuda):
    """
    The stub sampler. Return the original data.
    """
    trainadj_cache = {}
    if normalization in trainadj_cache:
        r_adj = trainadj_cache[normalization]
    else:
        r_adj = _preprocess_adj(normalization, train_adj, cuda)
        trainadj_cache[normalization] = r_adj
    fea = _preprocess_fea(train_features, cuda)
    return r_adj, fea


def randomedge_sampler(train_adj,train_features, percent, normalization, cuda):
    """
    Randomly drop edge and preserve percent% edges.
    """
    "Opt here"
    if percent >= 1.0:
        return stub_sampler(train_adj,train_features,normalization, cuda)

    nnz = train_adj.nnz
    perm = np.random.permutation(nnz)
    preserve_nnz = int(nnz * percent)
    perm = perm[:preserve_nnz]
    r_adj = sp.coo_matrix((train_adj.data[perm],
                           (train_adj.row[perm],
                            train_adj.col[perm])),
                          shape=train_adj.shape)
    r_adj = _preprocess_adj(normalization, r_adj, True)
    fea = _preprocess_fea(train_features, True)

    return r_adj, fea

def get_test_set(adj,features,normalization, cuda):
    """
    Return the test set.
    """

    r_adj = _preprocess_adj(normalization, adj, cuda)
    fea = _preprocess_fea(features, cuda)
    return r_adj, fea

def get_val_set(adj,features,normalization, cuda):
    """
    Return the validataion set. Only for the inductive task.
    Currently behave the same with get_test_set
    """
    return get_test_set(adj,features,normalization, cuda)
def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)

    features = row_normalize(features)
    return adj, features

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


# adapted from tkipf/gcn
def load_citation(dataset_str="cora", lbl_noise=0, seed=42):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

###
    adj, features = preprocess_citation(adj, features, "NoNorm")

####
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    features = normalize(features)
    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    labels = add_label_noise(idx_train, labels, lbl_noise, seed)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    adj = sys_normalized_adjacency(adj)#5/15
    adj = sparse_mx_to_torch_sparse_tensor(adj)#5/15
    return adj, features, labels, idx_train, idx_val, idx_test


# adapted from PetarV/GAT
def run_dfs(adj, msk, u, ind, nb_nodes):
    if msk[u] == -1:
        msk[u] = ind
        #for v in range(nb_nodes):
        for v in adj[u,:].nonzero()[1]:
            #if adj[u,v]== 1:
            run_dfs(adj, msk, v, ind, nb_nodes)

def dfs_split(adj):
    # Assume adj is of shape [nb_nodes, nb_nodes]
    nb_nodes = adj.shape[0]
    ret = np.full(nb_nodes, -1, dtype=np.int32)

    graph_id = 0

    for i in range(nb_nodes):
        if ret[i] == -1:
            run_dfs(adj, ret, i, graph_id, nb_nodes)
            graph_id += 1

    return ret

def test(adj, mapping):
    nb_nodes = adj.shape[0]
    for i in range(nb_nodes):
        #for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i] != mapping[j]:
              #  if adj[i,j] == 1:
                 return False
    return True

def find_split(adj, mapping, ds_label):
    nb_nodes = adj.shape[0]
    dict_splits={}
    for i in range(nb_nodes):
        #for j in range(nb_nodes):
        for j in adj[i, :].nonzero()[1]:
            if mapping[i]==0 or mapping[j]==0:
                dict_splits[0]=None
            elif mapping[i] == mapping[j]:
                if ds_label[i]['val'] == ds_label[j]['val'] and ds_label[i]['test'] == ds_label[j]['test']:

                    if mapping[i] not in dict_splits.keys():
                        if ds_label[i]['val']:
                            dict_splits[mapping[i]] = 'val'

                        elif ds_label[i]['test']:
                            dict_splits[mapping[i]]='test'

                        else:
                            dict_splits[mapping[i]] = 'train'

                    else:
                        if ds_label[i]['test']:
                            ind_label='test'
                        elif ds_label[i]['val']:
                            ind_label='val'
                        else:
                            ind_label='train'
                        if dict_splits[mapping[i]]!= ind_label:
                            print ('inconsistent labels within a graph exiting!!!')
                            return None
                else:
                    print ('label of both nodes different, exiting!!')
                    return None
    return dict_splits


def add_label_noise(idx_train, labels, noise_ratio, seed):
    if noise_ratio == None:
        return labels
    random.seed(seed)
    num_nodes = idx_train[-1]
    erasing_pool = torch.arange(num_nodes)
    print('pool', erasing_pool.shape)
    np.random.seed(seed)
    noise_num = int(num_nodes*noise_ratio)
    print('noise_num',noise_num)

    sele_idx = [j for j in random.sample(range(0,num_nodes), noise_num)]
    for i in sele_idx:
        re_lb = random.sample([j for j in range(max(labels)+1) if j != labels[idx_train[i]]], 1)
        labels[idx_train[i]] = re_lb[0]
    # import ipdb; ipdb.set_trace()
    return labels
