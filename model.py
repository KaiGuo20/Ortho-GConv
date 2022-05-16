import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
# import wandb
from torch.nn.parameter import Parameter
from layers import *
device = torch.device("cuda:0")
class GraphConvolution_(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Ortho_Trans(torch.nn.Module):
    def __init__(self, T=5, norm_groups=1, *args, **kwargs):
        super(Ortho_Trans, self).__init__()
        self.T = T
        self.norm_groups = norm_groups
        self.eps = 1e-5

    def matrix_power3(self, Input):
        B=torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  # type: torch.Tensor
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc, Zc.transpose(1, 2))
        eye = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        S = S + self.eps*eye
        norm_S = S.norm(p='fro', dim=(1, 2), keepdim=True)

        S = S.div(norm_S)
        B = [torch.Tensor([]) for _ in range(self.T + 1)]
        B[0] = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        for t in range(self.T):
            B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, self.matrix_power3(B[t]), S)
        W = B[self.T].matmul(Zc).div_(norm_S.sqrt())
        return W.view_as(weight)



class Ortho_GCN(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, T1, T2, group1, group2,weight_beta, Ortho, bias):
        super(Ortho_GCN, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution_Ortho(nhidden, nhidden,T2, group2,weight_beta, Ortho))
        self.fcs = nn.ModuleList()
        self.fcs.append(GraphConvolution_(nfeat, nhidden, bias))
        self.fcs.append(GraphConvolution_(nhidden, nclass, bias))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.nlayers = nlayers
        self.nhidden = nhidden
        if Ortho:
            self.weight_normalization1 = Ortho_Trans(T=T1, norm_groups=group1)
            self.fcs[0].weight.data = self.weight_normalization1(self.fcs[0].weight)
        self.gama = Parameter(torch.ones(nhidden, nhidden))

    def forward(self, x,adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training).cuda()
        layer_inner = self.act_fn(self.fcs[0](x,adj))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):

            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner, adj)

        return F.log_softmax(layer_inner, dim=1)

    def ortho_loss(self):

        loss = None
        I = torch.eye(self.nhidden).cuda()

        for i, layer in enumerate(self.convs):
            p = layer.t
            if loss is None:

                w = torch.mm(p.transpose(0,1), p) - torch.mm(self.gama, I)

                loss = w.pow(2).sum()
            else:
                w = torch.mm(p.transpose(0, 1), p) - torch.mm(self.gama, I)

                loss += w.pow(2).sum()

        return loss


class GraphConvolution_Ortho(nn.Module):

    def __init__(self, in_features, out_features,T2, group2,weight_beta, Ortho):
        super(GraphConvolution_Ortho, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.Ortho = Ortho
        self.weight_beta = weight_beta
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.weight_normalization = Ortho_Trans(T=T2, norm_groups=group2)
        self.reset_parameters()
        self.t = None
        self.we = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hi = torch.spmm(adj, input)
        if self.Ortho:
            I = torch.eye(self.in_features).cuda()
            self.we = self.weight_beta * self.weight.cuda() + (1 - self.weight_beta) * I
            self.t = self.weight_normalization(self.we)
            if self.training:
                self.t.retain_grad()
            output = torch.mm(hi, self.t)

        else:
            output = torch.mm(hi, self.weight)


        return output

if __name__ == '__main__':
    pass


class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, variant, T1, T2, group1, group2, weight_beta,Ortho):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(Ortho_GCNII(nhidden, nhidden,T2, group2,weight_beta,Ortho,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.nlayers = nlayers
        self.alpha = alpha
        self.lamda = lamda
        self.nhidden = nhidden
        if Ortho :
            self.weight_normalization1 = Ortho_Trans(T=T1, norm_groups=group1)
            self.fcs[0].weight.data = self.weight_normalization1(self.fcs[0].weight)
        self.gama = Parameter(torch.ones(nhidden, nhidden))

    def forward(self, x,adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training).cuda()

        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)

        return F.log_softmax(layer_inner, dim=1)

    def ortho_loss(self):

        loss = None
        I = torch.eye(self.nhidden).cuda()

        for i, layer in enumerate(self.convs):
            p = layer.t
            if loss is None:
                w = torch.mm(p.transpose(0,1), p) -  torch.mm(self.gama, I)
                loss = w.pow(2).sum()
            else:
                w = torch.mm(p.transpose(0, 1), p) -  torch.mm(self.gama, I)
                loss += w.pow(2).sum()

        return loss

class Ortho_GCNII(nn.Module):

    def __init__(self, in_features, out_features, T2, group2, weight_beta, Ortho, residual=False, variant=False):
        super(Ortho_GCNII, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.Ortho = Ortho
        self.weight_beta = weight_beta
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.weight_normalization = Ortho_Trans(T=T2, norm_groups=group2)#2，2 2,1            cora(5141)
        self.bn = torch.nn.BatchNorm1d(in_features)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.6)
        self.reset_parameters()
        self.t = None
        self.we = None

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support


        if self.Ortho:
            I = torch.eye(self.in_features).cuda()
            self.we = self.weight_beta * self.weight.cuda() + (1 - self.weight_beta) * I
            self.t = self.weight_normalization(self.we)
            
            if self.training:
                self.t.retain_grad()
            output = theta * torch.mm(support, self.t) + (1 - theta) * r
        else:
            output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+0.05*input
        return output



class JKNet_Model(nn.Module):

    def __init__(self,
                 nfeat,
                 nhid,
                 nclass,
                 nhidlayer,
                 dropout,
                 T1,
                 T2,
                 group1,
                 group2,
                 weight_beta,
                 Ortho,
                 baseblock="mutigcn",
                 inputlayer="gcn",
                 outputlayer="gcn",
                 nbaselayer=0,
                 activation=lambda x: x,
                 withbn=True,
                 withloop=True,
                 aggrmethod="add",
                 mixmode=False):

        super(JKNet_Model, self).__init__()
        self.mixmode = mixmode
        self.dropout = dropout
        self.nhid = nhid
        self.gama = Parameter(torch.ones(nhid, nhid))
        if baseblock == "densegcn":
            self.BASEBLOCK = DenseGCNBlock
        if inputlayer == "gcn":
            # input gc
            self.ingc = GraphConvolutionBS(nfeat, nhid, activation, withbn, withloop)
            if Ortho:
                self.weight_normalization1 = Ortho_Trans(T=T1, norm_groups=group1)  # 2 1 5
                self.ingc.weight.data = self.weight_normalization1(self.ingc.weight.data)
            baseblockinput = nhid
        outactivation = lambda x: x
        if outputlayer == "gcn":
            self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)


        self.midlayer = nn.ModuleList()

        for i in range(nhidlayer):
            gcb = self.BASEBLOCK(in_features=baseblockinput,
                                 out_features=nhid,
                                 nbaselayer=nbaselayer,
                                 withbn=withbn,
                                 withloop=withloop,
                                 activation=activation,
                                 dropout=dropout,
                                 dense=False,
                                 aggrmethod=aggrmethod,
                                 T2 = T2,
                                 group2 = group2,
                                 weight_beta = weight_beta,
                                 Ortho = Ortho)
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()
        # output gc
        outactivation = lambda x: x  # we donot need nonlinear activation here.
        self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)

        if mixmode:
            self.midlayer = self.midlayer.to(device)
            self.outgc = self.outgc.to(device)

    def forward(self, fea, adj):
        # input
        if self.mixmode:
            x = self.ingc(fea, adj.cpu())
        else:
            x = self.ingc(fea, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        if self.mixmode:
            x = x.to(device)
        for i in range(len(self.midlayer)):
            midgc = self.midlayer[i]
            x = midgc(x, adj)
        x = self.outgc(x, adj)
        x = F.log_softmax(x, dim=1)

        return x

    def ortho_loss(self):

        loss = None
        I = torch.eye(self.nhid).cuda()
        for i in range(len(self.midlayer)):
            for i, layer in enumerate(self.midlayer[i].model.hiddenlayers):
                p = layer.t
                if loss is None:
                    w = torch.mm(p.transpose(0,1), p) - torch.mm(self.gama, I)
                    loss = w.pow(2).sum()
                else:
                    w = torch.mm(p.transpose(0, 1), p) - torch.mm(self.gama, I)
                    loss += w.pow(2).sum()

        return loss










