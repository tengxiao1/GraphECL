import argparse

from model import LogReg, Model
from utils import load
from model_scal import Modelmini

import torch
import torch as th
import torch.nn as nn
import numpy as np
import warnings
import random
from sampler import NeighborSampler
import dgl

warnings.filterwarnings('ignore')
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
random.seed(1024)
torch.cuda.manual_seed_all(1024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description='SAGCL')

parser.add_argument('--dataname', type=str, default='comp', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--epochs', type=int, default=50, help='Training epochs.')
parser.add_argument('--lr1', type=float, default=5e-4, help='Learning rate of pretraining.')
parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate of linear evaluator.')
parser.add_argument('--wd1', type=float, default=1e-6, help='Weight decay of pretraining.')
parser.add_argument('--wd2', type=float, default=1e-5, help='Weight decay of linear evaluator.')
parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
parser.add_argument('--use_mlp', action='store_true', default=False, help='Use MLP instead of GNN')
parser.add_argument('--temp', type=float, default=0.5, help='Temperature hyperparameter.')
parser.add_argument("--hid_dim", type=int, default=2048, help='Hidden layer dim.')
parser.add_argument('--moving_average_decay', type=float, default=0.0)
parser.add_argument('--num_MLP', type=int, default=1)
parser.add_argument('--lambda_loss', type=float, default=1)
parser.add_argument('--lam', type=float, default=0.001)
parser.add_argument('--run_times', type=int, default=1)
# parser.add_argument('--neg_sample', type=int, default=0)
# parser.add_argument('--pos_sample', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--neg_size', type=int, default=1024)
parser.add_argument('--sample_neighbors', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--num_MLP_encoder', type=int, default=2)

args = parser.parse_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda'
else:
    args.device = 'cpu'


def write_results(acc):
    f = open("results_mini/" + args.dataname + '_heterSSL', 'a+')
    f.write(args.dataname + ' --epoch ' + str(args.epochs) + ' --lr1 ' + str(args.lr1) + ' --lambda_loss ' + str(
        args.lambda_loss) + ' --moving_average_decay ' + str(args.moving_average_decay) + ' --dimension ' + str(
        args.hid_dim) + ' --tau ' + str(args.temp) + ' --lam ' + str(args.lam) + ' --num_MLP ' + str(
        args.num_MLP) + ' --n_layers ' + str(
        args.n_layers) + ' --batch_size ' + str(
        args.batch_size) + ' --neg_size ' + str(
        args.neg_size) + ' --sample_neighbors ' + str(
        args.sample_neighbors) + ' --num_MLP_encoder ' + str(
        args.num_MLP_encoder) + f'   Final Test: {np.mean(acc):.4f} ± {np.std(acc):.4f}\n')
    f.close()


if __name__ == '__main__':
    print(args)
    # load hyperparameters
    dataname = args.dataname
    hid_dim = args.hid_dim
    out_dim = args.hid_dim
    n_layers = args.n_layers
    temp = args.temp
    epochs = args.epochs
    lr1 = args.lr1
    wd1 = args.wd1
    lr2 = args.lr2
    wd2 = args.wd2
    device = args.device

    graph, feat, labels, num_class, train_idx, val_idx, test_idx = load(dataname)
    all_idx = torch.arange(0, feat.shape[0])
    in_dim = feat.shape[1]
    weights = torch.ones(args.neg_size, dtype=float).to(device)
    # train_idx = train_idx.to(device)
    # print(all_idx.shape[0])
    model = Modelmini(in_dim, hid_dim, out_dim, n_layers, temp, weights, args.sample_neighbors, args.num_MLP_encoder, args.use_mlp, args.moving_average_decay,
                          args.num_MLP,
                          args.lambda_loss, args.lam)

    # model = Model(in_dim, hid_dim, out_dim, n_layers, temp, args.use_mlp, args.moving_average_decay, args.num_MLP,
    #               args.lambda_loss, args.lam)
    model = model.to(device)

    # graph = graph.to(device)
    feat = feat.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    graph = graph.remove_self_loop().add_self_loop()

    sampler = NeighborSampler(np.ones(args.n_layers, dtype=int) * args.sample_neighbors, args.neg_size, feat.shape[0])
    dataloader = dgl.dataloading.DataLoader(
        graph, all_idx, sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        # pin_memory=True,
        num_workers=args.num_workers)

    for epoch in range(epochs):
        # model.update_moving_average()
        model.train()
        optimizer.zero_grad()
        for input_nodes, output_nodes, blocks, neighbor_index, input_nodes_neg, output_nodes_neg, blocks_neg in dataloader:
            input_features = feat[input_nodes]
            input_features_neg = feat[input_nodes_neg]
            loss = model([b.to(device) for b in blocks], input_features, [b.to(device) for b in blocks_neg], input_features_neg, feat[neighbor_index])
            # # loss = model(graph, feat, neg_sample=args.neg_sample, pos_sample=args.pos_sample)
            loss.backward()
            optimizer.step()

        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

    print("=== Evaluation ===")
    graph = graph.remove_self_loop().add_self_loop()
    embeds = model.get_embedding(graph.to(device), feat)
    results = []
    for run in range(0, args.run_times):

        train_idx_tmp = train_idx
        val_idx_tmp = val_idx
        test_idx_tmp = test_idx
        train_embs = embeds[train_idx_tmp]
        val_embs = embeds[val_idx_tmp]
        test_embs = embeds[test_idx_tmp]

        label = labels.to(device)

        train_labels = label[train_idx_tmp]
        val_labels = label[val_idx_tmp]
        test_labels = label[test_idx_tmp]

        train_feat = feat[train_idx_tmp]
        val_feat = feat[val_idx_tmp]
        test_feat = feat[test_idx_tmp]

        ''' Linear Evaluation '''
        logreg = LogReg(train_embs.shape[1], num_class)
        opt = th.optim.Adam(logreg.parameters(), lr=lr2, weight_decay=wd2)

        logreg = logreg.to(device)
        loss_fn = nn.CrossEntropyLoss()

        best_val_acc = 0
        eval_acc = 0

        for epoch in range(600):
            logreg.train()
            opt.zero_grad()
            logits = logreg(train_embs)
            preds = th.argmax(logits, dim=1)
            train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]
            loss = loss_fn(logits, train_labels)
            loss.backward()
            opt.step()

            logreg.eval()
            with th.no_grad():
                val_logits = logreg(val_embs)
                test_logits = logreg(test_embs)

                val_preds = th.argmax(val_logits, dim=1)
                test_preds = th.argmax(test_logits, dim=1)

                val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    eval_acc = test_acc

                print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc,
                                                                                         test_acc))
        results.append(eval_acc.item())
        print(f'Validation Accuracy: {best_val_acc}, Test Accuracy: {eval_acc}')
    # write_results(results)