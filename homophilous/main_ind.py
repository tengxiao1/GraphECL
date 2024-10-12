import argparse

from model import LogReg, Model
from utils import load, graph_split

import torch
import torch as th
import torch.nn as nn
import numpy as np
import warnings
import random

seed = 1024

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
parser.add_argument('--split_rate', type=float, default=0.2)
parser.add_argument('--run_times', type=int, default=1)
parser.add_argument('--neg_sample', type=int, default=0)
parser.add_argument('--pos_sample', type=int, default=0)

args = parser.parse_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'


def write_results(acc, acc_trans):
    f = open("results_ind2/" + args.dataname + '_heterSSL', 'a+')
    f.write(args.dataname + ' --epoch ' + str(args.epochs) + ' --lr1 ' + str(args.lr1) + ' --lambda_loss ' + str(
        args.lambda_loss) + ' --moving_average_decay ' + str(args.moving_average_decay) + ' --dimension ' + str(
        args.hid_dim) + ' --tau ' + str(args.temp) + ' --lam ' + str(args.lam) + ' --num_MLP ' + str(
        args.num_MLP) + ' --n_layers ' + str(
        args.n_layers) + f'   Final Test: {np.mean(acc):.4f} ± {np.std(acc):.4f} Transductive Test: {np.mean(acc_trans):.4f} ± {np.std(acc_trans):.4f} Production: {0.8 * np.mean(acc_trans) + 0.2 * np.mean(acc):.4f}\n')
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

    graph, feat, labels, num_class, train_idx, val_idx, test_idx = load(dataname, inductive=True, seed=seed)
    idx_obs, test_idx_tran, test_idx = graph_split(train_idx, val_idx, test_idx, args.split_rate)
    obs_g = graph.subgraph(idx_obs)
    in_dim = feat.shape[1]

    model = Model(in_dim, hid_dim, out_dim, n_layers, temp, args.use_mlp, args.moving_average_decay, args.num_MLP,
                  args.lambda_loss, args.lam)
    model = model.to(device)

    obs_g = obs_g.to(device)
    feat = feat.to(device)
    obs_feat = feat[idx_obs]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    obs_g = obs_g.remove_self_loop().add_self_loop()
    for epoch in range(epochs):
        # model.update_moving_average()
        model.train()
        optimizer.zero_grad()

        loss = model(obs_g, obs_feat, neg_sample= args.neg_sample, pos_sample = args.pos_sample)
        loss.backward()
        optimizer.step()

        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

    print("=== Evaluation ===")
    obs_g = obs_g.remove_self_loop().add_self_loop()
    embeds = model.get_embedding(obs_g, feat)
    results = []
    results_trans = []
    for run in range(0, args.run_times):

        train_idx_tmp = train_idx
        val_idx_tmp = val_idx
        test_idx_tmp = test_idx
        train_embs = embeds[train_idx_tmp]
        val_embs = embeds[val_idx_tmp]
        test_embs = embeds[test_idx_tmp]
        test_embs_trans = embeds[test_idx_tran]

        label = labels.to(device)

        train_labels = label[train_idx_tmp]
        val_labels = label[val_idx_tmp]
        test_labels = label[test_idx_tmp]
        test_labels_trans = label[test_idx_tran]

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
        eval_acc_trans = 0

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
                test_logits_trans = logreg(test_embs_trans)

                val_preds = th.argmax(val_logits, dim=1)
                test_preds = th.argmax(test_logits, dim=1)
                test_preds_trans = th.argmax(test_logits_trans, dim=1)

                val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
                test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]
                test_acc_trans = th.sum(test_preds_trans == test_labels_trans).float() / test_labels_trans.shape[0]

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    eval_acc = test_acc
                    eval_acc_trans = test_acc_trans

                print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc_ind:{:4f}, test_acc_trans:{:4f}'.format(epoch, train_acc, val_acc,
                                                                                         test_acc, test_acc_trans))
        results.append(eval_acc.item())
        results_trans.append(eval_acc_trans.item())
        print(f'Validation Accuracy: {best_val_acc}, Test Accuracy: {eval_acc}')
    # write_results(results, results_trans)

