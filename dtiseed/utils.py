import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score,  precision_recall_curve
from sklearn.metrics import auc as auc3
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
from sklearn.metrics.pairwise import cosine_similarity as cos
import time
import scipy.spatial.distance as dist
from CLaugmentdti import *


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


default_configure = {
    'batch_size': 20
}

heter_configure = {
    "lr": 0.0001,
    "dropout": 0,
    "cl_loss_co": 0.5,
    "reg_co": 0.0003,
    "in_size": 512,
    "hidden_size": 256,
    "out_size": 128,
    "weight_decay": 1e-10

}



def setup(args,seed):
    args.update(default_configure)
    set_random_seed(seed)
    return args


def comp_jaccard(M):
    matV = np.mat(M)
    x = dist.pdist(matV, 'jaccard')

    k = np.eye(matV.shape[0])
    count = 0
    for i in range(k.shape[0]):
        for j in range(i + 1, k.shape[1]):
            k[i][j] = x[count]
            k[j][i] = x[count]
            count += 1
    return k


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def load_dataset(network_path):
    """
    meta_path of drug

    """
    drug_drug = np.loadtxt(network_path + 'mat_drug_drug.txt')

    drug_disease = np.loadtxt(network_path + 'mat_drug_disease.txt')
    disease_drug = drug_disease.T
    drug_sideeffect = np.loadtxt(network_path + 'mat_drug_se.txt')

    sideeffect_drug = drug_sideeffect.T
    "-o"
    drug_drug_protein = np.loadtxt(network_path + 'mat_drug_protein_o.txt')
    """
    meta_path of protein

    """
    protein_protein = np.loadtxt(network_path + 'mat_protein_protein.txt')
    protein_protein_drug = drug_drug_protein.T

    protein_disease = np.loadtxt(network_path + 'mat_protein_disease.txt')
    disease_protein = protein_disease.T

    d_d = dgl.graph(sparse.csr_matrix(drug_drug), ntype='drug', etype='dd')
    num_drug = d_d.number_of_nodes()

    d_di = dgl.bipartite(sparse.csr_matrix(drug_disease), 'drug', 'ddi', 'disease')
    di_d = dgl.bipartite(sparse.csr_matrix(disease_drug), 'disease', 'did', 'drug')
    d_d_p = dgl.bipartite(sparse.csr_matrix(drug_drug_protein), 'drug', 'ddp', 'protein')

    d_se = dgl.bipartite(sparse.csr_matrix(drug_sideeffect), 'drug', 'dse', 'sideeffect')
    se_d = dgl.bipartite(sparse.csr_matrix(sideeffect_drug), 'sideeffect', 'sed', 'drug')

    p_p = dgl.graph(sparse.csr_matrix(protein_protein), ntype='protein', etype='pp')
    num_protein = p_p.number_of_nodes()

    p_di = dgl.bipartite(sparse.csr_matrix(protein_disease), 'protein', 'pdi', 'disease')
    p_d_d = dgl.bipartite(sparse.csr_matrix(protein_protein_drug), 'protein', 'pdd', 'drug')

    di_p = dgl.bipartite(sparse.csr_matrix(disease_protein), 'disease', 'dip', 'protein')

    dg = dgl.hetero_from_relations([d_d, d_se, se_d, d_di, di_d, d_d_p, p_d_d])
    pg = dgl.hetero_from_relations([p_p, p_di, di_p, p_d_d, d_d_p])
    graph = [dg, pg]

    dti_o = np.loadtxt(network_path + 'mat_drug_protein.txt')
    train_positive_index = []
    whole_negative_index = []

    for i in range(np.shape(dti_o)[0]):
        for j in range(np.shape(dti_o)[1]):
            if int(dti_o[i][j]) == 1:
                train_positive_index.append([i, j])

            else:
                whole_negative_index.append([i, j])

    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size= len(train_positive_index),
                                             replace=False)


    data_set = np.zeros((len(negative_sample_index) +  len(train_positive_index), 3),
                        dtype=int)
    count = 0

    for i in train_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    f = open("dti_cledge.txt", "w", encoding="utf-8")

    for i in range(count):
        for j in range(count):
            if data_set[i][0] == data_set[j][0] or data_set[i][1] == data_set[j][1]:
                f.write(f"{i}\t{j}\n")

    for i in range(len(negative_sample_index)):
        data_set[count][0] = whole_negative_index[negative_sample_index[i]][0]
        data_set[count][1] = whole_negative_index[negative_sample_index[i]][1]
        data_set[count][2] = 0
        count += 1
    f = open(f"dti_index.txt", "w", encoding="utf-8")
    for i in data_set:
        f.write(f"{i[0]}\t{i[1]}\t{i[2]}\n")

    dateset = data_set
    f = open("dtiedge.txt", "w", encoding="utf-8")
    for i in range(dateset.shape[0]):
        for j in range(i, dateset.shape[0]):
            if dateset[i][0] == dateset[j][0] or dateset[i][1] == dateset[j][1]:
                f.write(f"{i}\t{j}\n")
    f.close()


    node_num = [num_drug, num_protein]
    all_meta_paths = [[['dd'], ['dse', 'sed'], ['ddi', 'did'], ['ddp', 'pdd']],
                      [['pp'], ['pdi', 'dip'], ['pdd', 'ddp']]]
    return dateset, graph, node_num, all_meta_paths


def load_graph(feature_edges, n):
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sparse.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(n, n),
                             dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sparse.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    return nfadj



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def construct_fgraph(features, topk):

    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)
    edge = []
    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                edge.append([i, vv])
    return edge


def generate_knn(data):
    """ top-k number"""
    topk = 3

    edge = construct_fgraph(data, topk)
    res = []

    for line in edge:
        start, end = line[0], line[1]
        if int(start) < int(end):
            res.append([start, end])
    return res


def constructure_graph(dateset, h1, h2, task="dti", aug=False):
    feature = torch.cat((h1[dateset[:, :1]], h2[dateset[:, 1:2]]), dim=2)

    feature = feature.squeeze(1)

    edge = np.loadtxt(f"{task}edge.txt", dtype=int)

    if aug:
        edge_aug = aug_random_edge(np.array(edge))
        edge_aug = load_graph(np.array(edge_aug), dateset.shape[0])
        edge = load_graph(np.array(edge), dateset.shape[0])

        feature_aug = aug_random_mask(feature)
        return edge, feature, edge_aug, feature_aug
    edge = torch.from_numpy(edge.T)
    edge1 = load_graph(np.array(edge), dateset.shape[0])

    return edge1,edge, feature


def constructure_knngraph(dateset, h1, h2, aug=False):
    feature = torch.cat((h1[dateset[:, :1]], h2[dateset[:, 1:2]]), dim=2)

    feature = feature.squeeze(1)

    fedge = np.array(generate_knn(feature.cpu().detach().numpy()))

    if aug:
        fedge_aug = aug_random_edge(np.array(fedge))
        feature_aug = aug_random_mask(feature)
        fedge_aug = load_graph(np.array(fedge_aug), dateset.shape[0])
        fedge = load_graph(np.array(fedge), dateset.shape[0])

        return fedge, feature, fedge_aug, feature_aug
    else:
        fedge = torch.from_numpy(fedge.T)
        fedge1 = load_graph(np.array(fedge), dateset.shape[0])

        return fedge1,fedge, feature


def get_clGraph(data, task):
    cledg = np.loadtxt(f"{task}_cledge.txt", dtype=int)

    cl = torch.eye(len(data))
    for i in cledg:
        cl[i[0]][i[1]] = 1
    return cl


def get_set(data, split=5):
    """
    :param data: dataset and label
    :return:
    testset index and trainset index
    """
    set1 = []
    set2 = []
    skf = StratifiedKFold(n_splits=split, shuffle=True)
    for train_index, test_index in skf.split(data[:, :2], data[:, 2:3]):
        set1.append(train_index)
        set2.append(test_index)
    return set1[0].reshape(-1), set2[0].reshape(-1)

def get_cross(data, split=5):
    """
    :param data: dataset and label
    :return:
    testset index and trainset index
    """
    set1 = []
    set2 = []
    skf = StratifiedKFold(n_splits=split, shuffle=True)
    for train_index, test_index in skf.split(data[:, :2], data[:, 2:3]):
        set1.append(train_index)
        set2.append(test_index)
    return set1, set2


def get_roc(out, label):
    return roc_auc_score(label.cpu(), out[:, 1:].cpu().detach().numpy())

def get_pr(out, label):
    precision, recall, thresholds = precision_recall_curve(label.cpu(), out[:, 1:].cpu().detach().numpy())
    return auc3(recall, precision)


def get_f1score(out, label):
    return f1_score(label.cpu(), out.argmax(dim=1).cpu().detach().numpy())


def get_L2reg(parameters):
    reg = 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg


