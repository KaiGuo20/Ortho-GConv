import argparse
import sys
import time

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.utils import convert
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from layer import  GCNIIdenseConv, GCNConv1
from torch_geometric.nn import GCNConv
import scipy.sparse as sp
import torch_geometric.transforms as T


import math
import random
import numpy as np


parser = argparse.ArgumentParser(description='OGBN-Arxiv (Full-Batch)')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')  # 42 41（86.3）
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--log_steps', type=int, default=100)
parser.add_argument('--num_layers', type=int, default=16)  # 16
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (L2 loss on parameters).')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=1000)  # 1000
parser.add_argument('--runs', type=int, default=10)  # 10
parser.add_argument('--patience', type=int, default=200, help='patience')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
parser.add_argument('--norm', default='bn', help='norm layer.')
parser.add_argument('--T1', type=int, default=2, help='number of first newton iteration')
parser.add_argument('--T2', type=int, default=2, help='number of second newton iteration')
parser.add_argument('--group1', type=int, default=1, help='number of norm group')
parser.add_argument('--group2', type=int, default=2, help='number of norm group')
parser.add_argument('--weight_beta', type=float, default=0.1, help='weight_beta.')  # 0.5
parser.add_argument('--Ortho', default=False, help='use or not orthogonalization.')
parser.add_argument('--sparse', default=False, help='use or not sparse.')
parser.add_argument('--model', default='GCNIIdense_model', help='GCNNet')
parser.add_argument('--gama', type=float, default=0.0001, help='loss')
parser.add_argument('--data', default='arxiv', help='arxiv, proteins')
class Logger(object):
    def __init__(self, filename='arxiv-2layer-ortho.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(stream=sys.stdout)
# args = parser.parse_args()
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

##############################################

####################
if args.data == "arxiv":
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    if args.sparse:
        data = T.ToSparseTensor()(data.to(device))
    else :
        data = data.to(device)
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    train_idx = split_idx['train'].to(device)


if args.data == "arxiv":
    evaluator = Evaluator(name='ogbn-arxiv')



class GCNNet(nn.Module):
    # def __init__(self, dataset, hidden=256, num_layers=3):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,  T1, T2, group1,
                 group2, weight_beta, Ortho):
        """
        :param dataset: 数据集
        :param hidden: 隐藏层维度，默认256
        :param num_layers: 模型层数，默认为3
        """
        super(GCNNet, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.Gconvs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.hidden = hidden_channels

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.gama = Parameter(torch.ones(hidden_channels, hidden_channels))

        for i in range(self.num_layers - 2):
            self.Gconvs.append(GCNConv1(hidden_channels, hidden_channels,T1,T2, group1,group2,weight_beta,Ortho))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for Gconv in self.convs:
            Gconv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        # x, adj_t = data.x, data.adj_t
        if args.sparse:
            x, edge_index = data.x, data.adj_t
        else :
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.convs[0](x, edge_index)
        x = self.bns[0](x)  # 小数据集不norm反而效果更好
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        for i in range(self.num_layers - 2):
            x = self.Gconvs[i](x, edge_index)
            x = self.bns[i+1](x)  # 小数据集不norm反而效果更好
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = F.log_softmax(x, dim=1)

        return x
    def ortho_loss(self):

        loss = None
        I = torch.eye(self.hidden).cuda()

        for i, layer in enumerate(self.Gconvs):
            p = layer.weight
            if loss is None:
                w = torch.mm(p.transpose(0,1), p) -  torch.mm(self.gama, I)
                loss = w.pow(2).sum()
            else:
                w = torch.mm(p.transpose(0, 1), p) -  torch.mm(self.gama, I)
                loss += w.pow(2).sum()

        return loss

class GCNIIdense_model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,dropout,alpha,norm,T1, T2, group1, group2, weight_beta,Ortho):
        super(GCNIIdense_model, self).__init__()
        self.in_channels = in_channels
        self.convs = torch.nn.ModuleList()
        self.linear = torch.nn.ModuleList()
        self.linear.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers):
            self.convs.append(GCNIIdenseConv(hidden_channels, hidden_channels,T1,T2, group1,group2,weight_beta,Ortho,bias=norm))
        self.linear.append(torch.nn.Linear(hidden_channels,out_channels))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.dropout = dropout
        self.alpha = alpha
        self.hidden = hidden_channels
        self.gama = Parameter(torch.ones(hidden_channels, hidden_channels))

    def forward(self,data):
        if args.sparse:
            x, edge_index = data.x, data.adj_t
        else :
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        _hidden = []
        x = F.dropout(x, self.dropout ,training=self.training)
        x = F.relu(self.linear[0](x))
        _hidden.append(x)
        for i,con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, self.dropout ,training=self.training)
            x = F.relu(con(x, edge_index,self.alpha, _hidden[0],edge_weight))+_hidden[-1]
            _hidden.append(x)
        x = F.dropout(x, self.dropout ,training=self.training)
        x = self.linear[-1](x)

        return F.log_softmax(x, dim=1)

    def ortho_loss(self):

        loss = None
        I = torch.eye(self.hidden).cuda()

        for i, layer in enumerate(self.convs):
            p = layer.weight1
            if loss is None:
                w = torch.mm(p.transpose(0,1), p) -  torch.mm(self.gama, I)
                loss = w.pow(2).sum()
            else:
                w = torch.mm(p.transpose(0, 1), p) -  torch.mm(self.gama, I)
                loss += w.pow(2).sum()

        return loss



def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()

    pred = model(data)[train_idx]

    # import ipdb; ipdb.set_trace()

    loss = F.nll_loss(pred, data.y.squeeze(1)[train_idx])

    # if args.Ortho == "True":
    #     loss += args.gama * model.ortho_loss()#0.0001

    # if args.Ortho:
    #     loss += args.gama * model.ortho_loss()#0.0001
    loss.backward()
    optimizer.step()

    return loss.item()



@torch.no_grad()
def test(model, data, y_true,split_idx, evaluator):
    model.eval()

    out = model(data)
    if args.data == "proteins":
        y_pred = out
    elif args.data == "arxiv":
        y_pred = out.argmax(dim=-1, keepdim=True)


    # import  ipdb; ipdb.set_trace()
    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    acc_list = []
    run_time = []
    for run in range(args.runs):
        if args.model == "GCNIIdense_model":
            num_classes = dataset.num_classes
            # import ipdb; ipdb.set_trace()
            model = GCNIIdense_model(data.x.size(-1), args.hidden_channels,
                        num_classes, args.num_layers,
                        args.dropout, args.alpha, args.norm, args.T1, args.T2, args.group1, args.group2,
                        args.weight_beta, args.Ortho).to(device)
        elif args.model == "GCNNet":

            num_classes = dataset.num_classes
            model = GCNNet(data.x.size(-1), args.hidden_channels,
                                     num_classes, args.num_layers,
                                      args.T1, args.T2, args.group1, args.group2,
                                     args.weight_beta, args.Ortho).to(device) #####dat
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        bad_counter = 0
        best_val = 0
        final_test_acc = 0
        t_total = time.time()
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, data.y,split_idx, evaluator)
            train_acc, valid_acc, test_acc = result
            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
            if valid_acc > best_val:
                best_val = valid_acc
                final_test_acc = test_acc
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                break
        acc_list.append(final_test_acc*100)
        print(run+1,':',acc_list[-1])
        run_time.append(time.time() - t_total)
        print("Train cost: {:.4f}s".format(time.time() - t_total))
    acc_list=torch.tensor(acc_list)
    print('---all results',acc_list)
    print(f'Avg Test: {acc_list.mean():.2f} ± {acc_list.std():.2f}')
    print('average epoch train time', np.mean(run_time))
    



if __name__ == "__main__":
    main()
