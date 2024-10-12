from heterophilous.layers import MLP, MLP_generator, PairNorm, FNN
from heterophilous.utils import  contrastive_loss
import torch
import torch.nn as nn
from dgl.nn import  GraphConv
import random
import copy
from torch.nn.functional import normalize
import numpy as np
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)

        x = self.act_fn(x)
        x = self.layer2(x)

        return x
def update_moving_average(target_ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = target_ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# Main Autoencoder structure here
class GNNStructEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, layer_num, sample_size, tau, lam=0.1, norm_mode="PN-SCS", norm_scale=20, lambda_loss=1,moving_average_decay=0.0, num_MLP=3):

        super(GNNStructEncoder, self).__init__()
        self.norm = PairNorm(norm_mode, norm_scale)
        self.out_dim = hidden_dim
        self.lambda_loss = lambda_loss
        self.tau = tau
        self.lam = lam
        # GNN Encoder
        self.layer_num = layer_num
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_dim, hidden_dim))
        for i in range(layer_num - 1):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))
        # self.graphconv1 = GraphConv(in_dim, hidden_dim)
        # self.graphconv2 = GraphConv(hidden_dim, hidden_dim)


        # self.target_graphconv1 = copy.deepcopy(self.graphconv1)
        # self.target_graphconv2 = copy.deepcopy(self.graphconv2)

        # set_requires_grad(self.target_graphconv1, False)
        # set_requires_grad(self.target_graphconv2, False)
        self.target_ema_updater = EMA(moving_average_decay)
        # self.feature_decoder = FNN(hidden_dim, hidden_dim, in_dim, 3)
        self.encoder_target=MLP(in_dim, hidden_dim, hidden_dim, use_bn=True)
        self.num_MLP = num_MLP
        if num_MLP != 0:
            self.projector = MLP_generator(hidden_dim, hidden_dim, num_MLP)
        self.in_dim = in_dim
        self.sample_size = sample_size

    # def update_moving_average(self):
    #     # assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
    #     assert self.target_graphconv1 or self.target_graphconv2 is not None, 'target encoder has not been created yet'
    #     update_moving_average(self.target_ema_updater, self.target_graphconv1, self.graphconv1)
    #     update_moving_average(self.target_ema_updater, self.target_graphconv2, self.graphconv2)

    def forward_encoder(self, g, h):
        GNN_emb = self.convs[0](g, h)
        for i in range(1, self.layer_num):
            GNN_emb = self.convs[i](g, GNN_emb)
        GNN_emb=normalize(GNN_emb, p=2, dim=-1)
        MLP_emb = normalize(self.encoder_target(g,h), p=2, dim=-1)
        if self.num_MLP == 0:
            projected_rep = normalize(MLP_emb, p=2, dim=-1)
        else:
            projected_rep = normalize(self.projector(MLP_emb), p=2, dim=-1)

        return GNN_emb, MLP_emb, projected_rep

    def get_emb(self, neighbor_indexes, gt_embeddings):
        sampled_embeddings = []
        if len(neighbor_indexes) < self.sample_size:
            sample_indexes = neighbor_indexes
            sample_indexes += np.random.choice(neighbor_indexes, self.sample_size - len(sample_indexes)).tolist()
        else:
            sample_indexes = random.sample(neighbor_indexes, self.sample_size)

        for index in sample_indexes:
            sampled_embeddings.append(gt_embeddings[index])

        return torch.stack(sampled_embeddings)

    # Sample neighbors from neighbor set, if the length of neighbor set less than sample size, then do the padding.
    def sample_neighbors(self, neighbor_dict,GNN_emb):
        sampled_embeddings_list = []
        sampled_embeddings_neg_list = []

        for index, embedding in enumerate(GNN_emb):
            neighbor_indexes = neighbor_dict[index]
            sampled_embeddings = self.get_emb(neighbor_indexes, GNN_emb)
            sampled_embeddings_list.append(sampled_embeddings)
            sampled_neg_embeddings = self.get_emb(range(0, len(neighbor_dict)), GNN_emb)
            sampled_embeddings_neg_list.append(sampled_neg_embeddings)
            # sampled_neg_MLP_embeddings = self.get_emb(range(0, len(neighbor_dict)), projected_rep)
            # sampled_embeddings_MLP_neg_list.append(sampled_neg_MLP_embeddings)


        return torch.stack(sampled_embeddings_list), torch.stack(sampled_embeddings_neg_list)

    def reconstruction_neighbors(self, GNN_emb, projected_rep, neighbor_dict):
        sampled_embeddings_list, sampled_embeddings_neg_list = self.sample_neighbors(neighbor_dict, GNN_emb)
        projected_rep = projected_rep.unsqueeze(1)
        neighbor_recons_loss = contrastive_loss(projected_rep, GNN_emb, sampled_embeddings_list, sampled_embeddings_neg_list ,self.tau, self.lambda_loss, self.lam)
        return neighbor_recons_loss


    def neighbor_decoder(self, GNN_emb, projected_rep, neighbor_dict):
        neighbor_recons_loss = self.reconstruction_neighbors(GNN_emb,projected_rep, neighbor_dict)
        loss = neighbor_recons_loss
        return loss


    def forward(self, g, h, neighbor_dict, device):
        GNN_emb, MLP_emb, projected_rep = self.forward_encoder(g, h)
        loss = self.neighbor_decoder(GNN_emb, projected_rep,neighbor_dict)
        return loss, MLP_emb