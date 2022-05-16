from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from networkx.drawing.tests.test_pylab import plt
from torch.distributions import Beta
# from torch_geometric.datasets import Planetoid, CitationFull,WikiCS, Coauthor, Amazon
# import torch_geometric.transforms as T
from utils import *
from model import *
import uuid
import os.path as osp
# import wandb
import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0, help='weight decay (L2 loss on convs parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on fcs parameters).')
parser.add_argument('--layer', type=int, default=32, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=1500, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--model', default='GCNII', help='GCNII, Orhto_GCN, JKNet')
parser.add_argument('--Ortho', default=False, help='use or not orthogonalization.')
parser.add_argument('--bias',  default=False, help='use or not bias.')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')
parser.add_argument('--T1', type=int, default=2, help='number of first newton iteration')
parser.add_argument('--T2', type=int, default=2, help='number of second newton iteration')
parser.add_argument('--group1', type=int, default=1, help='number of norm group')
parser.add_argument('--group2', type=int, default=2, help='number of norm group')
parser.add_argument('--gama', type=float, default=0.0001, help='loss')
parser.add_argument('--get', default='loss', help='loss, acc')
parser.add_argument('--weight_beta', type=float, default=0.1, help='weight_beta.')
parser.add_argument('--weight_decay', type=float, default=0.005, help='weight decay (L2 loss on parameters).')
parser.add_argument("--normalization", default="AugNormAdj",
                    help="The normalization on the adj matrix.")
parser.add_argument("--sampling_percent", type=float, default=0.5,
                    help="The percent of the preserve edges. If it equals 1, no sampling is done on adj matrix.")
parser.add_argument('--no_cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--lal_rate', type=float, default=0, help='label noise ratio')#0.1

# Training JKNet parameter
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Disable validation during training.')
parser.add_argument('--lradjust', action='store_true',
                    default=False, help='Enable leraning rate adjust.(ReduceLROnPlateau or Linear Reduce)')
parser.add_argument("--mixmode", action="store_true",
                    default=False, help="Enable CPU GPU mixing mode.")
parser.add_argument("--warm_start", default="",
                    help="The model name to be loaded for warm start.")
parser.add_argument('--debug', action='store_true',
                    default=False, help="Enable the detialed training output.")
parser.add_argument('--dataset', default="cora", help="The data set")
parser.add_argument('--datapath', default="data/", help="The data path.")
parser.add_argument("--early_stopping", type=int,
                    default=0, help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
parser.add_argument("--no_tensorboard", default=False, help="Disable writing logs to tensorboard")

# JKNet Model parameter
parser.add_argument('--type',
                    help="Choose the model to be trained.(mutigcn, resgcn, densegcn, inceptiongcn)")
parser.add_argument('--inputlayer', default='gcn',
                    help="The input layer of the model.")
parser.add_argument('--outputlayer', default='gcn',
                    help="The output layer of the model.")
parser.add_argument('--withbn', action='store_true', default=False,
                    help='Enable Bath Norm GCN')
parser.add_argument('--withloop', action="store_true", default=False,
                    help="Enable loop layer GCN")
parser.add_argument('--nhiddenlayer', type=int, default=1,
                    help='The number of hidden layers.')
parser.add_argument("--baseblock", default="res", help="The base building block (resgcn, densegcn, mutigcn, inceptiongcn).")
parser.add_argument("--nbaseblocklayer", type=int, default=1,
                    help="The number of layers in each baseblock")
parser.add_argument("--aggrmethod", default="default",
                    help="The aggrmethod for the layer aggreation. The options includes add and concat. Only valid in resgcn, densegcn and inecptiongcn")
parser.add_argument("--task_type", default="full", help="The node classification task type (full and semi). Only valid for cora, citeseer and pubmed dataset.")
# import pdb; pdb.set_trace()



args = parser.parse_args()
if args.aggrmethod == "default":
    if args.type == "resgcn":
        args.aggrmethod = "add"
    else:
        args.aggrmethod = "concat"
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
print(cudaid,checkpt_file)
adj, features, labels,idx_train,idx_val,idx_test = load_citation(args.data,args.lal_rate, args.seed)
features = features.to(device)
adj = adj.to(device)

def train():

    model.train()
    optimizer.zero_grad()
    output = model(features,adj)#features
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    if args.Ortho:
        loss_train += args.gama * model.ortho_loss()#0.0001
    loss_train.backward()

    optimizer.step()
    # import ipdb; ipdb.set_trace()
    return loss_train.item(),acc_train.item()


def validate():
    model.eval()
    with torch.no_grad():
        output = model(features,adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test():
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test.item(),acc_test.item()


ACC = []
run_time =[]

for i in range(10):
    if args.model == "Ortho_GCN":
        model = Ortho_GCN(nfeat=features.shape[1],
                          nlayers=args.layer,
                          nhidden=args.hidden,
                          nclass=int(labels.max()) + 1,
                          dropout=args.dropout,
                          T1=args.T1,
                          T2=args.T2,
                          group1=args.group1,
                          group2=args.group2,
                          weight_beta=args.weight_beta,
                          Ortho = args.Ortho,
                          bias = args.bias).to(device)
        optimizer = optim.Adam([
            {'params': model.params1, 'weight_decay': args.wd1},
            {'params': model.params2, 'weight_decay': args.wd2},
        ], lr=args.lr)
    elif args.model == "GCNII" :
        model = GCNII(nfeat=features.shape[1],
                          nlayers=args.layer,
                          nhidden=args.hidden,
                          nclass=int(labels.max()) + 1,
                          dropout=args.dropout,
                          lamda=args.lamda,
                          alpha=args.alpha,
                          variant=args.variant,
                          T1=args.T1,
                          T2=args.T2,
                          group1=args.group1,
                          group2=args.group2,
                          weight_beta=args.weight_beta,
                          Ortho = args.Ortho).to(device)
        optimizer = optim.Adam([
            {'params': model.params1, 'weight_decay': args.wd1},
            {'params': model.params2, 'weight_decay': args.wd2},
        ], lr=args.lr)
    elif args.model == "JKNet":
        model = JKNet_Model(nfeat=features.shape[1],
                         nhid=args.hidden,
                         nclass=int(labels.max()) + 1,
                         nhidlayer=args.nhiddenlayer,
                         dropout=args.dropout,
                         baseblock=args.type,
                         inputlayer=args.inputlayer,
                         outputlayer=args.outputlayer,
                         nbaselayer=args.nbaseblocklayer,
                         activation=F.relu,
                         withbn=args.withbn,
                         withloop=args.withloop,
                         aggrmethod=args.aggrmethod,
                         mixmode=args.mixmode,
                         T1 = args.T1,
                         T2 = args.T2,
                         group1 = args.group1,
                         group2 = args.group2,
                         weight_beta = args.weight_beta,
                         Ortho = args.Ortho).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("model error.")

    t_total = time.time()
    bad_counter = 0
    best = 999999999
    best1 = 0
    best_epoch = 0
    acc = 0
    train_time = []
    for epoch in range(args.epochs):
        t = time.time()
        loss_tra,acc_tra = train()
        epoch_time = time.time()-t
        train_time.append(epoch_time)


        loss_val,acc_val = validate()
        if(epoch+1)%1 == 0:
            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                'acc:{:.2f}'.format(acc_tra*100),
                '| val',
                'loss:{:.3f}'.format(loss_val),
                'acc:{:.2f}'.format(acc_val*100))
        if args.get == "loss":
            if loss_val < best :
                best = loss_val
                best1 = acc_val-0.2
                best_epoch = epoch
                acc = acc_val
                torch.save(model.state_dict(), checkpt_file)
                bad_counter = 0
            else:
                bad_counter += 1

        if args.get == "acc":
            if acc_val > best1:

                best1 = acc_val
                best_epoch = epoch
                acc = acc_val
                torch.save(model.state_dict(), checkpt_file)
                bad_counter = 0
            else:
                bad_counter += 1

        if bad_counter == args.patience:
            break

    if args.test:
        acc = test()[1]
    ACC.append(acc*100)

    run_time.append(time.time() - t_total)
    print("Train cost: {:.4f}s".format(time.time() - t_total))
    print('Load {}th epoch'.format(best_epoch))
    print("Test" if args.test else "Val","acc.:{:.1f}".format(acc*100))
    print(ACC)

print('---all results',ACC)
print(np.mean(ACC))
print(np.std(ACC, ddof=1))







