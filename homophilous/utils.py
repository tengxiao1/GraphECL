import numpy as np
import torch as th

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset
from torch_geometric.datasets import WebKB, Actor
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
import networkx as nx
import dgl
import torch
import random
import torch.nn.functional as F
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from ogb.nodeproppred import DglNodePropPredDataset

np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
random.seed(1024)


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

def binarize_labels(labels, sparse_output=False, return_classes=False):
    if hasattr(labels[0], "__iter__"):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix

def load(name, inductive=False, seed = 0, labelrate_train=20, labelrate_val=30):

    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'comp':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'cs':
        dataset = CoauthorCSDataset()
    elif name == 'physics':
        dataset = CoauthorPhysicsDataset()
    elif name =='arxiv':
        dataset = DglNodePropPredDataset('ogbn-arxiv', './data/')
    elif name == 'products':
        dataset = DglNodePropPredDataset('ogbn-products', './data/')
    elif name == 'proteins':
        dataset = DglNodePropPredDataset('ogbn-proteins', './data/')
    elif name == 'mag':
        dataset = DglNodePropPredDataset('ogbn-mag', './data/')

    OGB_data = ["arxiv", "products", 'proteins', 'mag']

    if inductive and name not in OGB_data:

        graph = dataset[0]
        random_state = np.random.RandomState(seed)


        train_idx, val_idx, test_idx = get_train_val_test_split(
            random_state, binarize_labels(graph.ndata['label'].numpy()), labelrate_train, labelrate_val
        )
        train_idx = th.LongTensor(train_idx)
        val_idx = th.LongTensor(val_idx)
        test_idx = th.LongTensor(test_idx)
        num_class = dataset.num_classes
    else:
        citegraph = ['cora', 'citeseer', 'pubmed']
        cograph = ['photo', 'comp', 'cs', 'physics']
        if name in citegraph:
            graph = dataset[0]
            train_mask = graph.ndata.pop('train_mask')
            val_mask = graph.ndata.pop('val_mask')
            test_mask = graph.ndata.pop('test_mask')

            train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
            val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
            test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

            num_class = dataset.num_classes

        if name in cograph:
            graph = dataset[0]
            train_ratio = 0.1
            val_ratio = 0.1
            test_ratio = 0.8

            N = graph.number_of_nodes()
            train_num = int(N * train_ratio)
            val_num = int(N * (train_ratio + val_ratio))

            idx = np.arange(N)
            np.random.shuffle(idx)

            train_idx = idx[:train_num]
            val_idx = idx[train_num:val_num]
            test_idx = idx[val_num:]

            train_idx = th.LongTensor(train_idx)
            val_idx = th.LongTensor(val_idx)
            test_idx = th.LongTensor(test_idx)
            num_class = dataset.num_classes

        if name in OGB_data:
            graph, labels = dataset[0]
            labels = labels.squeeze()
            splitted_idx = dataset.get_idx_split()
            train_idx, val_idx, test_idx = (
                splitted_idx["train"],
                splitted_idx["valid"],
                splitted_idx["test"],
            )
            if dataset == "ogbn-arxiv":
                srcs, dsts = graph.all_edges()
                graph.add_edges(dsts, srcs)
                graph = graph.remove_self_loop().add_self_loop()

            num_class = dataset.num_classes


    # if name in hetergraph:
    #     data = torch_dataset[0]
    #     data.edge_index = to_undirected(data.edge_index)
    #     G = nx.from_edgelist(data.edge_index.transpose(0, 1).numpy().tolist())
    #     graph = dgl.from_networkx(G)
    #     graph.ndata['feat'] = data.x
    #     data.train_mask = data.train_mask.transpose(0, 1)
    #     data.val_mask = data.val_mask.transpose(0, 1)
    #     data.test_mask = data.test_mask.transpose(0, 1)
    #     train_idx = []
    #     val_idx = []
    #     test_idx = []
    #     for i in range(0, len(data.train_mask)):
    #         train_idx.append(torch.where(data.train_mask[i])[0])
    #         val_idx.append(torch.where(data.val_mask[i])[0])
    #         test_idx.append(torch.where(data.test_mask[i])[0])
    #
    #     labels = data.y
    #     graph.ndata['label'] = labels
    #     num_class = int(max(labels)) + 1


    feat = graph.ndata.pop('feat')
    if name not in OGB_data:
        labels = graph.ndata.pop('label')
    print(labels.shape)

    return graph, feat, labels, num_class, train_idx, val_idx, test_idx
