import sys
sys.path.append("..")
from heterophilous import utils2
import dgl
import torch
from heterophilous.model import GNNStructEncoder
import numpy as np
from tqdm import tqdm
from layers import LogReg
import statistics
import argparse
import random
from utils import NodeClassificationDataset, graph_split
import torch.nn.functional as F


# Training
def train(g, feats, feats_all, lr, epoch, device, lambda_loss, hidden_dim, sample_size=10, moving_average_decay=0.0):
    in_nodes, out_nodes = g.edges()
    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())

    # sample_size = max(neighbor_num_list)
    in_dim = feats.shape[1]
    GNNModel = GNNStructEncoder(in_dim, hidden_dim, args.num_GNN, sample_size, lambda_loss=lambda_loss, tau=args.tau,
                                lam=args.lam, moving_average_decay=moving_average_decay, num_MLP=args.num_MLP)
    GNNModel.to(device)

    opt = torch.optim.Adam([{'params': GNNModel.parameters()}], lr=lr, weight_decay=0.0003)

    for i in tqdm(range(epoch)):
        feats = feats.to(device)
        loss, node_embeddings = GNNModel(g, feats, neighbor_dict, device=device)
        opt.zero_grad()
        loss.backward()
        print(i, loss.item())
        opt.step()
        # GNNModel.update_moving_average()
    node_embeddings = F.normalize(GNNModel.encoder_target(g, feats_all), p=2, dim=-1)
    return node_embeddings.cpu().detach(), loss.item()


def evaluate(model, loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels)
    return (correct / total).item()


def write_results(acc, acc_trans, best_epoch):
    best_epoch = [str(tmp) for tmp in best_epoch]
    f = open("results_ind2/" + args.dataset + '_heterSSL', 'a+')
    f.write(args.dataset + ' --epochs ' + str(args.epoch_num) + ' --seed ' + str(args.seed) + ' --lr ' + str(
        args.lr) + ' --lambda_loss ' + str(args.lambda_loss) + ' --moving_average_decay ' + str(
        args.moving_average_decay) + ' --dimension ' + str(args.dimension) + ' --sample_size ' + str(
        args.sample_size) + ' --wd2 ' + str(args.wd2) + ' --num_MLP ' + str(args.num_MLP) + ' --num_GNN ' + str(
        args.num_GNN) + ' --tau ' + str(args.tau) + ' --lam ' + str(args.lam) + ' --best_epochs ' + " ".join(
        best_epoch) + f'   Final Test: {np.mean(acc):.4f} ± {np.std(acc):.4f} Transductive Test: {np.mean(acc_trans):.4f} ± {np.std(acc_trans):.4f} Production: {0.8 * np.mean(acc_trans) + 0.2 * np.mean(acc):.4f}\n')
    f.close()


def train_new_datasets(dataset_str, epoch_num=10, lr=5e-6, lambda_loss=1, sample_size=10, hidden_dim=None,
                       moving_average_decay=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g, labels, split_lists = utils2.read_real_datasets(dataset_str)
    g = g.to(device)
    node_features = g.ndata['attr']
    node_labels = labels
    node_labels = node_labels.to(device)
    # attr, feat
    if hidden_dim == None:
        hidden_dim = node_features.shape[1]
    else:
        hidden_dim = hidden_dim
    acc = []
    acc_trans = []
    epochs = []
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    for index in range(10):
        idx_obs, idx_test_tran, idx_test_ind = graph_split(split_lists[index]['train_idx'], split_lists[index]['valid_idx'], split_lists[index]['test_idx'], rate=args.split_rate)
        idx_obs = idx_obs.to(device)
        obs_g = g.subgraph(idx_obs)
        obs_feat = node_features[idx_obs]
        obs_g = dgl.add_self_loop(obs_g)
        node_embeddings, loss = train(obs_g, obs_feat, node_features, lr=lr, epoch=epoch_num, device=device,
                                      lambda_loss=lambda_loss, sample_size=sample_size, hidden_dim=hidden_dim,
                                      moving_average_decay=moving_average_decay)
        input_dims = node_embeddings.shape
        print(input_dims[1])
        class_number = int(max(node_labels)) + 1
        FNN = LogReg(input_dims[1], class_number).to(device)
        FNN = FNN.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(FNN.parameters(), lr=1e-2, weight_decay=args.wd2)
        node_embeddings = node_embeddings.to(device)
        dataset = NodeClassificationDataset(node_embeddings, node_labels)

        split = utils2.DataSplit(dataset, split_lists[index]['train_idx'], split_lists[index]['valid_idx'],
                                 idx_test_ind, shuffle=True)

        train_loader, val_loader, test_loader = split.get_split(batch_size=100000, num_workers=0)
        # train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
        best = -float('inf')
        best_epoch = 0
        test_acc = 0
        test_acc_trans = 0
        for epoch in range(3000):
            for i, data in enumerate(train_loader, 0):
                # data = data.to(device)
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                y_pred = FNN(inputs)
                loss = criterion(y_pred, labels)
                # train_loss = loss
                # print(epoch, i, loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in val_loader:
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = FNN(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)
                        total += labels.size(0)
                        correct += torch.sum(predicted == labels)
                if correct / total > best:
                    best = correct / total
                    test_acc = evaluate(FNN, test_loader)
                    best_epoch = epoch

                    outputs = FNN(node_embeddings[idx_test_tran])
                    _, predicted = torch.max(outputs.data, 1)
                    correct = torch.sum(predicted == node_labels[idx_test_tran])
                    test_acc_trans = (correct / len(idx_test_tran)).item()
                    # print(test_acc_trans)
                    # torch.save(FNN.state_dict(), 'best_mlp_{}.pkl'.format(index))
                # print(str(epoch), correct / total)
        # with torch.no_grad():
        #     FNN.load_state_dict(torch.load('best_mlp_{}.pkl'.format(index)))
        #     correct = 0
        #     total = 0
        #     for data in test_loader:
        #         inputs, labels = data
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)
        #         outputs = FNN(inputs)
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += torch.sum(predicted == labels)
        print(test_acc)
        acc.append(test_acc)
        acc_trans.append(test_acc_trans)
        epochs.append(best_epoch)
    print("mean:")
    print(statistics.mean(acc))
    print("std:")
    print(statistics.stdev(acc))
    write_results(acc, acc_trans, epochs)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset', type=str, default="texas")
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--epoch_num', type=int, default=50)
    parser.add_argument('--lambda_loss', type=float, default=1)
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--dimension', type=int, default=512)
    parser.add_argument('--moving_average_decay', type=float, default=0.0)
    parser.add_argument('--identify', type=str, default="sample")
    parser.add_argument('--dataset_type', type=str, default="real")
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--wd2', type=float, default=1e-2)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--num_MLP', type=int, default=1)
    parser.add_argument('--num_GNN', type=int, default=2)
    parser.add_argument('--split_rate', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=2014)

    args = parser.parse_args()
    # train_synthetic_graphs()™

    dataset_str = args.dataset
    train_new_datasets(dataset_str=dataset_str, lr=args.lr, epoch_num=args.epoch_num, lambda_loss=args.lambda_loss,
                       sample_size=args.sample_size, hidden_dim=args.dimension,
                       moving_average_decay=args.moving_average_decay)
