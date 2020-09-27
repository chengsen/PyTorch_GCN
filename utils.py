import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from dgl.data import CiteseerGraphDataset
from dgl.data import CoraGraphDataset
from dgl.data import PubmedGraphDataset


# def parse_index_file(filename):
#     """Parse index file."""
#     index = []
#     for line in open(filename):
#         index.append(int(line.strip()))
#     return index


def load_data(dataset):
    if dataset == 'cora':
        data = CoraGraphDataset()
    elif dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif dataset == 'pubmed':
        data = PubmedGraphDataset()

    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    nxg = g.to_networkx()
    adj = nx.to_scipy_sparse_matrix(nxg, dtype=np.float)

    adj = preprocess_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels, train_mask, val_mask, test_mask


# def preprocess_features(features):
#     """
#     Row-normalize feature matrix and convert to tuple representation
#     """
#     rowsum = np.array(features.sum(1))  # get sum of each row, [2708, 1]
#     r_inv = np.power(rowsum, -1).flatten()  # 1/rowsum, [2708]
#     r_inv[np.isinf(r_inv)] = 0.  # zero inf data
#     r_mat_inv = sp.diags(r_inv)  # sparse diagonal matrix, [2708, 2708]
#     features = r_mat_inv.dot(features)  # D^-1:[2708, 2708]@X:[2708, 2708]
#     return sparse_to_tuple(features)  # [coordinates, data, shape], []


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion
    to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


# def sparse_to_tuple(sparse_mx):
#     """
#     Convert sparse matrix to tuple representation.
#     """
#
#     def to_tuple(mx):
#         if not sp.isspmatrix_coo(mx):
#             mx = mx.tocoo()
#         coords = np.vstack((mx.row, mx.col)).transpose()
#         values = mx.data
#         shape = mx.shape
#         return coords, values, shape
#
#     if isinstance(sparse_mx, list):
#         for i in range(len(sparse_mx)):
#             sparse_mx[i] = to_tuple(sparse_mx[i])
#     else:
#         sparse_mx = to_tuple(sparse_mx)
#
#     return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  # D^-0.5AD^0.5


# def normalize(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx


def accuracy(pred, targ):
    pred = torch.max(pred, 1)[1]
    ac = ((pred == targ).float()).sum().item() / targ.size()[0]
    return ac


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
