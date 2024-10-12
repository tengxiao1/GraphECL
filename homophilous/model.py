import dgl.sampling
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GraphConv
import copy
import random

sigma = 1e-6
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
random.seed(1024)
  
torch.cuda.manual_seed_all(1024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


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

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, use_ln=False):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))
        self.use_ln = use_ln
        self.lns = nn.ModuleList()

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            for i in range(n_layers - 1):
                self.lns.append(nn.BatchNorm1d(hid_dim))
                # self.lns.append(nn.LayerNorm(hid_dim))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x):
        for i in range(self.n_layers - 1):
            if not self.use_ln:
                x = F.relu(self.convs[i](graph, x))
            else:
                x = F.relu(self.lns[i](self.convs[i](graph, x)))

        x = self.convs[-1](graph, x)

        return x

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new.t

def update_moving_average(target_ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = target_ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

class MLP_generator(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super(MLP_generator, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, output_dim))
        for layer in range(num_layers - 1):
            self.linears.append(nn.Linear(output_dim, output_dim))
        self.num_layers = num_layers
        # self.linear4 = nn.Linear(output_dim, output_dim)

    def forward(self, embedding):
        h = embedding
        for layer in range(self.num_layers - 1):
            h = F.relu(self.linears[layer](h))
        neighbor_embedding = self.linears[self.num_layers - 1](h)
        # neighbor_embedding = self.linear4(neighbor_embedding)
        return neighbor_embedding

def udf_u_add_log_e(edges):
    return {'m': torch.log(edges.dst['neg_sim'] + edges.src['neg_sim2'] )}

def udf_u_add_log_e2(edges):
    # print(edges.dst['sample'].shape)
    # print(edges.src['z'].shape)
    sim = torch.bmm(edges.dst['sample'], edges.src['z'].unsqueeze(1).transpose(1, 2)).squeeze().sum(-1)
    # print(sim)
    return {'m': torch.log(sim + edges.src['neg_sim2'] )}

# def udf_u_add_log_e2(edges):
#     return {'m': torch.log(edges.dst['neg_sim2'] )}

class Model(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, num_layers, temp, use_mlp=False, moving_average_decay=1.0, num_MLP=1,
                 lambda_loss=1, lam=0.001):
        super(Model, self).__init__()
        if use_mlp:
            self.encoder = MLP(in_dim, hid_dim, out_dim, use_bn=True)
        else:
            self.encoder = GCN(in_dim, hid_dim, out_dim, num_layers)
            # self.encoder = MLP(in_dim, hid_dim, out_dim, use_bn=True)
            self.encoder_target = MLP(in_dim, hid_dim, out_dim, use_bn=True)
            # self.encoder_target = copy.deepcopy(self.encoder)
            # set_requires_grad(self.encoder_target, False)
        self.temp = temp
        self.out_dim = out_dim
        self.lambda_loss = lambda_loss
        self.lam = lam
        self.target_ema_updater = EMA(moving_average_decay)
        self.num_MLP = num_MLP
        if num_MLP != 0:
            self.projector = MLP_generator(hid_dim, hid_dim, num_MLP)

    def get_embedding(self, graph, feat):
        # get embeddings from the MLP for evaluation
        trans_feature = self.encoder_target(graph, feat)
        return trans_feature.detach()

    def pos_score(self, graph, h, trans_feature, pos_sample=0):
        graph = graph.remove_self_loop().add_self_loop()
        graph.ndata['z'] = F.normalize(h, dim=-1)
        if self.num_MLP != 0:
            graph.ndata['q'] = F.normalize(self.projector(trans_feature))
        else:
            graph.ndata['q'] = F.normalize(trans_feature)
        if pos_sample == 0:
            graph.apply_edges(fn.u_mul_v('z', 'q', 'sim'))
            graph.edata['sim'] = graph.edata['sim'].sum(1) / self.temp
            # graph.edata['sim'] = torch.exp((graph.edata['sim'].sum(1)) / self.temp)
            graph.update_all(fn.copy_e('sim', 'm'), fn.mean('m', 'pos'))
            pos_score = graph.ndata['pos']
        else:
            sub_g = dgl.sampling.sample_neighbors(graph, torch.arange(0, h.shape[0]).cuda(), pos_sample)
            sub_g.apply_edges(fn.u_mul_v('z', 'q', 'sim'))
            sub_g.edata['sim'] = sub_g.edata['sim'].sum(1) / self.temp
            sub_g.update_all(fn.copy_e('sim', 'm'), fn.mean('m', 'pos'))
            pos_score = sub_g.ndata['pos']
            graph.ndata['pos'] = sub_g.ndata['pos']

        return pos_score, graph

    def sample_emb(self, z, trans_feature, neg_sample):
        sampled_embeddings_list_z = []
        sampled_embeddings_list_trans = []
        neighbor_indexes = range(0, z.shape[0])
        for i in range(0, z.shape[0]):
            sample_indexes = random.sample(neighbor_indexes, neg_sample)
            sampled_embeddings_list_z.append(z[sample_indexes])
            sampled_embeddings_list_trans.append(trans_feature[sample_indexes])
        return torch.stack(sampled_embeddings_list_z), torch.stack(sampled_embeddings_list_trans)

    def neg_score(self, graph, h, trans_feature, neg_sample=0):
        z = F.normalize(h, dim=-1)
        if self.num_MLP != 0:
            trans_feature = F.normalize(self.projector(trans_feature))
        else:
            trans_feature = F.normalize(trans_feature)
        # graph.edata['sim'] = torch.exp(graph.edata['sim'])
        if neg_sample == 0:
            neg_sim = torch.exp(torch.mm(z, z.t()) / self.temp)  # (i,j**)
            neg_sim2 = torch.exp(torch.mm(z, trans_feature.t()) / self.temp)  # (i**,j)
            neg_score = neg_sim.sum(1)
            graph.ndata['neg_sim'] = neg_score

            neg_score2 = neg_sim2.sum(1)
            graph.ndata['neg_sim2'] = self.lam * neg_score2

            graph.update_all(udf_u_add_log_e, fn.mean('m', 'neg'))
            neg_score = graph.ndata['neg']
        else:
            # print("Sampling")
            z_sample, z_trans_feature = self.sample_emb(z, trans_feature, neg_sample)
            # neg_sim = torch.exp(torch.bmm(z.unsqueeze(1), z_sample.transpose(-1, -2)).squeeze() / self.temp)
            neg_sim2 = torch.exp(torch.bmm(z.unsqueeze(1), z_trans_feature.transpose(-1, -2)).squeeze() / self.temp)
            graph.ndata['sample'] = z_sample
            # neg_score = neg_sim.sum(1)
            # graph.ndata['neg_sim'] = neg_score

            neg_score2 = neg_sim2.sum(1)
            graph.ndata['neg_sim2'] = self.lam * neg_score2

            graph.update_all(udf_u_add_log_e2, fn.mean('m', 'neg'))
            neg_score = graph.ndata['neg']


        return neg_score

    def update_moving_average(self):
        # assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.encoder_target is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.encoder_target, self.encoder)

    def forward(self, graph, feat, neg_sample=0, pos_sample=0):
        h = self.encoder(graph, feat)
        trans_feature = self.encoder_target(graph, feat)
        pos_score, graph = self.pos_score(graph, h, trans_feature, pos_sample)
        neg_score = self.neg_score(graph, h, (trans_feature), neg_sample)
        loss = (- pos_score + self.lambda_loss * neg_score).mean()
        # loss = (- pos_score ).mean()
        return loss





# class Model_stop(nn.Module):
#     # moving_average_decay=0
#     def __init__(self, in_dim, hid_dim, out_dim, num_layers, temp, use_mlp=False, moving_average_decay=0, num_MLP=1, lambda_loss=1):
#         super(Model_stop, self).__init__()
#         # self.encoder = MLP(in_dim, hid_dim, out_dim, use_bn=True)
#         # self.encoder_target = GCN(in_dim, hid_dim, out_dim, num_layers)
#         self.encoder = MLP(in_dim,hid_dim,out_dim, use_bn=True)
#         self.encoder_target = GCN(in_dim, hid_dim, out_dim, num_layers)
#         set_requires_grad(self.encoder_target, False)


#         self.temp = temp
#         self.out_dim = out_dim
#         self.lambda_loss = lambda_loss
#         self.target_ema_updater = EMA(moving_average_decay)
#         self.num_MLP = num_MLP
#         if num_MLP!=0:
#             self.projector = MLP_generator(hid_dim, hid_dim, num_MLP)


#     def get_embedding(self, graph, feat):
#         # get embeddings from the model for evaluation
#         h = self.encoder(graph,feat)
#         return h.detach()

#     def pos_score(self, graph, h, trans_feature):
#         graph = graph.remove_self_loop().add_self_loop()
#         # normalize trans_feature(lower branch) and store as 'z' node feature of graph
#         graph.ndata['z'] = F.normalize(h, dim=-1)
#         if self.num_MLP != 0:
#             # store above branch feature and store as 'q' node feature of graph
#             graph.ndata['q'] = F.normalize(self.projector(trans_feature))
#         else:
#             graph.ndata['q'] = F.normalize(trans_feature)
#         # edges.src['z']*edges.dst['q'] -> edata['sim']
#         graph.apply_edges(fn.u_mul_v('z', 'q', 'sim'))
#         # sum in dimension of d -> n_edge*1
#         # p^tu/T
#         graph.edata['sim'] = graph.edata['sim'].sum(1) / self.temp
#         # graph.edata['sim'] = torch.exp((graph.edata['sim'].sum(1)) / self.temp)
#         graph.update_all(fn.copy_e('sim', 'm'), fn.mean('m', 'pos'))
#         # pos_score=mean of similarity between 1. centre node in node feature 'z' 
#         # 2. neighbour(node has edge linked to center node) of centre node in node feature 'q'
#         pos_score = graph.ndata['pos']

#         return pos_score, graph

#     # def udf_u_add_log_e(edges):
#     #     return {'m': torch.log(edges.dst['neg_sim'] + edges.data['sim'])}

#     def neg_score(self, h, graph, rff_dim=None):
#         # above branch
#         z = F.normalize(h, dim=-1)
#         # exp(p^tu/T)
#         graph.edata['sim'] = torch.exp(graph.edata['sim'])
        
#         neg_sim = torch.exp(torch.mm(z, z.t()) / self.temp)
#         # sum(exp(v^t*v/T))
#         neg_score = neg_sim.sum(1)
#         # sum(exp(v^t*v/T))
#         graph.ndata['neg_sim'] = neg_score
#         graph.update_all(udf_u_add_log_e, fn.mean('m', 'neg'))
#         neg_score = graph.ndata['neg']
#         return neg_score

#     def update_moving_average(self):
#         # assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
#         assert self.encoder_target is not None, 'target encoder has not been created yet'
#         update_moving_average(self.target_ema_updater, self.encoder_target, self.encoder)

#     def forward(self, graph, feat):
#         # for name, param in self.encoder.named_parameters():
#         #     print(name, param.data.shape)
#         # print('---------------')
#         # for name, param in self.encoder_target.named_parameters():
#         #     print(name, param.data.shape)
#         h = self.encoder(graph,feat)
#         trans_feature = self.encoder_target(graph, feat)
#         pos_score, graph = self.pos_score(graph, h, trans_feature)
#         neg_score = self.neg_score(h, graph)
#         loss = (- pos_score + self.lambda_loss * neg_score).mean()
#         return loss