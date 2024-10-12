### Random tools useful for saveing stuff and manipulating pickle/numpy objects
import numpy as np
import pickle
import gzip
import re
import networkx as nx
import dgl
import torch


import logging
from functools import lru_cache

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
# import build_graph
from torch_geometric.datasets import WebKB, Actor
from torch_geometric.utils import to_undirected

import torch
import torch_geometric.transforms as T
from heterophilous.dataset import WikipediaNetwork
class DataSplit:

    def __init__(self, dataset, train_ind, val_ind, test_ind, shuffle=True):
        self.train_indices = train_ind
        self.val_indices = val_ind
        self.test_indices = test_ind
        self.dataset = dataset

        self.train_sampler = SubsetRandomSampler(self.train_indices)
        self.val_sampler = SubsetRandomSampler(self.val_indices)
        self.test_sampler = SubsetRandomSampler(self.test_indices)

    def get_train_split_point(self):
        return len(self.train_sampler) + len(self.val_indices)

    def get_validation_split_point(self):
        return len(self.train_sampler)

    @lru_cache(maxsize=4)
    def get_split(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train-validation-test dataloaders')
        self.train_loader = self.get_train_loader(batch_size=batch_size, num_workers=num_workers)
        self.val_loader = self.get_validation_loader(batch_size=batch_size, num_workers=num_workers)
        self.test_loader = self.get_test_loader(batch_size=batch_size, num_workers=num_workers)
        return self.train_loader, self.val_loader, self.test_loader

    @lru_cache(maxsize=4)
    def get_train_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing train dataloader')
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.train_sampler, shuffle=False, num_workers=num_workers)
        return self.train_loader

    @lru_cache(maxsize=4)
    def get_validation_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing validation dataloader')
        self.val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.val_sampler, shuffle=False, num_workers=num_workers)
        return self.val_loader

    @lru_cache(maxsize=4)
    def get_test_loader(self, batch_size=50, num_workers=4):
        logging.debug('Initializing test dataloader')
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=self.test_sampler, shuffle=False, num_workers=num_workers)
        return self.test_loader


def save_obj(obj, name, path, compress=False):
    # print path+name+ ".pkl"
    if compress is False:
        with open(path + name + ".pkl", 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with gzip.open(path + name + '.pklz','wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name,compressed=False):
    if compressed is False:
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        with gzip.open(name,'rb') as f:
            return pickle.load(f)


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(l):
    '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        float regex comes from https://stackoverflow.com/a/12643073/190597
        '''
    t = np.array([ int(re.split(r"([a-zA-Z]*)([0-9]*)", c)[2]) for c in l  ])
    order = np.argsort(t)
    return [l[o] for o in order]


def saveNet2txt(G, colors=[], name="net", path="plots/"):
    '''saves graph to txt file (for Gephi plotting)
    INPUT:
    ========================================================================
    G:      nx graph
    colors: colors of the nodes
    name:   name of the file
    path:   path of the storing folder
    OUTPUT:
    ========================================================================
    2 files containing the edges and the nodes of the corresponding graph
    '''
    if len(colors) == 0:
        colors = range(nx.number_of_nodes(G))
    graph_list_rep = [["Id","color"]] + [[i,colors[i]]
                      for i in range(nx.number_of_nodes(G))]
    np.savetxt(path + name + "_nodes.txt", graph_list_rep, fmt='%s %s')
    edges = G.edges(data=False)
    edgeList = [["Source", "Target"]] + [[v[0], v[1]] for v in edges]
    np.savetxt(path + name + "_edges.txt", edgeList, fmt='%s %s')
    print ("saved network  edges and nodes to txt file (for Gephi vis)")
    return


def read_real_datasets(datasets):
    if datasets in ["cornell", "texas", "wisconsin"]:
        torch_dataset = WebKB(root=f'../datasets_new/', name=datasets,
                          transform=T.NormalizeFeatures())
    elif datasets in ['squirrel', 'chameleon']:
        torch_dataset = WikipediaNetwork(root=f'../datasets_new/', name=datasets, geom_gcn_preprocess=True)
    elif datasets in ['crocodile']:
        torch_dataset = WikipediaNetwork(root=f'../datasets_new/', name=datasets, geom_gcn_preprocess=False)
    elif datasets == 'film':
        torch_dataset = Actor(root=f'../datasets_new/film/', transform=T.NormalizeFeatures())
    data = torch_dataset[0]
    data.edge_index = to_undirected(data.edge_index)
    G = nx.from_edgelist(data.edge_index.transpose(0, 1).numpy().tolist())
    g = dgl.from_networkx(G)
    g.ndata['attr'] = data.x
    data.train_mask = data.train_mask.transpose(0, 1)
    data.val_mask = data.val_mask.transpose(0, 1)
    data.test_mask = data.test_mask.transpose(0, 1)
    split_list = []
    for i in range(0, len(data.train_mask)):
        split_list.append({'train_idx': torch.where(data.train_mask[i])[0],
                           'valid_idx': torch.where(data.val_mask[i])[0],
                           'test_idx': torch.where(data.test_mask[i])[0]})
    labels = data.y

    return g, labels, split_list


