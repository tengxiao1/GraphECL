import sys
sys.path.append("..")
# from data import build_graph, utils
# import seaborn as sb
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import Dataset
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn as sk
import networkx as nx
import torch.nn.functional as F

def idx_split(idx, ratio):
    n = len(idx)
    cut = int(n * ratio)
    idx_idx_shuffle = torch.randperm(n)

    idx1_idx, idx2_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
    idx1, idx2 = idx[idx1_idx], idx[idx2_idx]
    # assert((torch.cat([idx1, idx2]).sort()[0] == idx.sort()[0]).all())
    return idx1, idx2

def graph_split(idx_train, idx_val, idx_test, rate):
    idx_test_ind, idx_test_tran = idx_split(idx_test, rate)

    idx_obs = torch.cat([idx_train, idx_val, idx_test_tran])
    # N1, N2 = idx_train.shape[0], idx_val.shape[0]
    # obs_idx_all = torch.arange(idx_obs.shape[0])
    # obs_idx_train = obs_idx_all[:N1]
    # obs_idx_val = obs_idx_all[N1 : N1 + N2]
    # obs_idx_test = obs_idx_all[N1 + N2 :]

    return idx_obs, idx_test_tran, idx_test_ind

def sample_per_class(
    random_state, labels, num_examples_per_class, forbidden_indices=None
):
    """
    Used in get_train_val_test_split, when we try to get a fixed number of examples per class
    """

    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [
            random_state.choice(
                sample_indices_per_class[class_index],
                num_examples_per_class,
                replace=False,
            )
            for class_index in range(len(sample_indices_per_class))
        ]
    )


def get_train_val_test_split(
    random_state,
    labels,
    train_examples_per_class=None,
    val_examples_per_class=None,
    test_examples_per_class=None,
    train_size=None,
    val_size=None,
    test_size=None,
):

    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))
    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False
        )

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state,
            labels,
            val_examples_per_class,
            forbidden_indices=train_indices,
        )
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(
            random_state,
            labels,
            test_examples_per_class,
            forbidden_indices=forbidden_indices,
        )
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert (
            len(np.concatenate((train_indices, val_indices, test_indices)))
            == num_samples
        )

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices

def mse_loss(predictions, targets):
    total_loss = torch.sqrt((predictions - targets).pow(2).mean(-1).mean(-1))
    return torch.sum(total_loss)


def contrastive_loss(projected_rep, GNN_emb, sampled_pos_GNN_list, sampled_neg_GNN_list, tau, lambda_loss, lam):
    # GNN_emb = GNN_emb.unsqueeze(1)

    # predictions MLP (NXD)  targets_pos GNN (Nx10xD)   output_emb GNN (NxD)   targets_neg GNN (NX10xD)    sampled_embeddings_MLP_neg_list MLP (NX10xD)
    pos = torch.exp(torch.bmm(projected_rep, sampled_pos_GNN_list.transpose(-1, -2)).squeeze() / tau)
    # neg1=torch.sum(torch.exp(torch.bmm(projected_rep, sampled_neg_GNN_list.transpose(-1, -2)).squeeze()/tau), dim=1).unsqueeze(-1)
    # neg2=torch.sum(torch.exp(torch.bmm((sampled_pos_GNN_list), sampled_neg_GNN_list.transpose(-1, -2)).squeeze()/tau), dim=-1)
    neg_score = torch.log(
        lam * torch.sum(torch.exp(torch.bmm(projected_rep, sampled_neg_GNN_list.transpose(-1, -2)).squeeze() / tau),
                        dim=1).unsqueeze(-1)
        + torch.sum(
            torch.exp(torch.bmm((sampled_pos_GNN_list), sampled_neg_GNN_list.transpose(-1, -2)).squeeze() / tau),
            dim=-1))
    # neg_score = torch.log(lam * torch.sum(torch.exp(torch.bmm(GNN_emb, sampled_embeddings_neg_list.transpose(-1, -2)).squeeze()/tau), dim=1).unsqueeze(-1)
    #                      + torch.sum(torch.exp(torch.bmm((sampled_embeddings_list), sampled_embeddings_neg_list.transpose(-1, -2)).squeeze()/tau), dim=-1))
    neg_score = torch.sum(neg_score, dim=1)
    pos_socre = torch.sum(torch.log(pos), dim=1)
    total_loss = torch.sum(lambda_loss * neg_score - pos_socre)
    total_loss = total_loss / sampled_pos_GNN_list.shape[0] / sampled_pos_GNN_list.shape[1]

    return total_loss

class NodeClassificationDataset(Dataset):
    def __init__(self, node_embeddings, labels):
        self.len = node_embeddings.shape[0]
        self.x_data = node_embeddings
        self.y_data = labels.long()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len




def unsupervised_evaluate(colors, labels_pred, trans_data, nb_clust):
    ami = sk.metrics.adjusted_mutual_info_score(colors, labels_pred)
    sil = sk.metrics.silhouette_score(trans_data, labels_pred, metric='euclidean')
    ch = sk.metrics.calinski_harabasz_score(trans_data, labels_pred)
    hom = sk.metrics.homogeneity_score(colors, labels_pred)
    comp = sk.metrics.completeness_score(colors, labels_pred)
    #print('Homogeneity \t Completeness \t AMI \t nb clusters \t CH \t  Silhouette \n')
    #print(str(hom) + '\t' + str(comp) + '\t' + str(ami) + '\t' + str(nb_clust) + '\t' + str(ch) + '\t' + str(sil))
    return hom, comp, ami, nb_clust, ch, sil




def average(lst):
    return sum(lst) / len(lst)


def write_graph2edgelist(G, role_id, filename):
    nx.write_edgelist(G, "{}.edgelist".format(filename), data=False)
    with open("{}.roleid".format(filename), "w") as f:
        for id in role_id:
            f.write(str(id) + "\n")

def set_pca( pca, embeddings):
    node_embedded = StandardScaler().fit_transform(embeddings)
    pca.fit(node_embedded)
    return pca
