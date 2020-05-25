import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(adj_file='adjacent_matrix.npy', ftr_lbl_file='feature_matrix.npy'):
    features_labels = np.load('feature_matrix.npy')
    features = sp.csr_matrix(features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(features_labels[:, -1])
    adj = sp.csr_matrix(np.load('adjacent_matrix.npy'), dtype=np.float32)
    adj += sp.eye(adj.shape[0])
    features = normalize(features)
    adj = laplacianize(adj)

    N = adj.shape[0]
    train = range(round(N * 0.5))
    val = range(round(N * 0.5), round(N * 0.75))
    test = range(round(N * 0.75), N)

    idx_train = train
    idx_val = val
    idx_test = test

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def laplacianize(mx):
    """Create laplacian matrix"""
    rowsum = np.array(mx.sum(1))
    r_sqrt_inv = np.power(rowsum, -1/2).flatten()
    r_sqrt_inv[np.isinf(r_sqrt_inv)] = 0.
    r_mat_sqrt_inv = sp.diags(r_sqrt_inv)
    mx = r_mat_sqrt_inv.dot(mx).dot(r_mat_sqrt_inv)
    return mx


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
