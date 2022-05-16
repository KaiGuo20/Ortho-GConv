import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import nn
import torch.nn.functional as F


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

class GraphConvolutionBS(Module):

    def __init__(self, in_features, out_features, activation=lambda x: x, withbn=True, withloop=True, bias=False,
                 res=False):

        super(GraphConvolutionBS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.res = res

        # Parameter setting.
        self.weight =  Parameter(torch.FloatTensor(in_features, out_features))
        # Is this the best practice or not?
        if withloop:
            self.self_weight = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.register_parameter("self_weight", None)

        if withbn:
            self.bn = torch.nn.BatchNorm1d(out_features)
        else:
            self.register_parameter("bn", None)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.self_weight is not None:
            stdv = 1. / math.sqrt(self.self_weight.size(1))
            self.self_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        # Self-loop
        if self.self_weight is not None:
            output = output + torch.mm(input, self.self_weight)

        if self.bias is not None:
            output = output + self.bias
        # BN
        if self.bn is not None:
            output = self.bn(output)
        # Res
        if self.res:
            return self.sigma(output) + input
        else:
            return self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolutionBS_ortho(Module):
    """
    GCN Layer with BN, Self-loop and Res connection.
    """

    def __init__(self, in_features, out_features, T2, group2, weight_beta, Ortho, activation=lambda x: x, withbn=True, withloop=True, bias=False,
                 res=False):

        super(GraphConvolutionBS_ortho, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.res = res
        self.Ortho = Ortho
        self.weight_beta = weight_beta
        # Parameter setting.
        self.register_parameter(
            "weight", Parameter(torch.FloatTensor(in_features, out_features))
        )
        self.weight_normalization = Ortho_Trans(T=T2, norm_groups=group2)

        if withloop:
            self.self_weight = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.register_parameter("self_weight", None)

        if withbn:
            self.bn = torch.nn.BatchNorm1d(out_features)
        else:
            self.register_parameter("bn", None)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.self_weight is not None:
            stdv = 1. / math.sqrt(self.self_weight.size(1))
            self.self_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        if self.Ortho:
            I = torch.eye(self.in_features).cuda()
            self.we = self.weight_beta * self.weight.cuda() + (1 - self.weight_beta) * I

            self.t = self.weight_normalization(self.we)
            if self.training:
                self.t.retain_grad()
            support = torch.mm(input, self.t)

        else:
            support = torch.mm(input, self.we)
        output = torch.spmm(adj, support)

        # Self-loop
        if self.self_weight is not None:
            output = output + torch.mm(input, self.self_weight)

        if self.bias is not None:
            output = output + self.bias
        # BN
        if self.bn is not None:
            output = self.bn(output)
        # Res
        if self.res:
            return self.sigma(output) + input
        else:
            return self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class GraphBaseBlock(Module):

    def __init__(self, in_features, out_features, nbaselayer, T2, group2,weight_beta,Ortho,
                 withbn=True, withloop=True, activation=F.relu, dropout=True,
                 aggrmethod="concat", dense=False):

        super(GraphBaseBlock, self).__init__()
        self.in_features = in_features
        self.hiddendim = out_features
        self.nhiddenlayer = nbaselayer
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dense = dense
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop
        self.Ortho = Ortho
        self.T2 = T2
        self.group2 = group2
        self.weight_beta = weight_beta
        self.hiddenlayers = nn.ModuleList()
        self.__makehidden()

        if self.aggrmethod == "concat" and dense == False:
            self.out_features = in_features + out_features
        elif self.aggrmethod == "concat" and dense == True:
            self.out_features = in_features + out_features * nbaselayer
        elif self.aggrmethod == "add":
            if in_features != self.hiddendim:
                raise RuntimeError("The dimension of in_features and hiddendim should be matched in add model.")
            self.out_features = out_features
        elif self.aggrmethod == "nores":
            self.out_features = out_features
        else:
            raise NotImplementedError("The aggregation method only support 'concat','add' and 'nores'.")

    def __makehidden(self):
        for i in range(self.nhiddenlayer):
            if i == 0:
                if self.Ortho:
                    layer = GraphConvolutionBS_ortho(self.in_features, self.hiddendim, self.T2, self.group2, self.weight_beta, self.Ortho,
                                                     self.activation, self.withbn, self.withloop)
                else:

                    layer = GraphConvolutionBS(self.in_features, self.hiddendim, self.activation, self.withbn,
                                                     self.withloop)
            else:
                if self.Ortho:
                    layer = GraphConvolutionBS_ortho(self.hiddendim, self.hiddendim, self.T2, self.group2, self.weight_beta, self.Ortho,
                                                     self.activation, self.withbn, self.withloop )
                else:

                    layer = GraphConvolutionBS(self.hiddendim, self.hiddendim, self.activation, self.withbn, self.withloop)
            self.hiddenlayers.append(layer)

    def _doconcat(self, x, subx):
        if x is None:
            return subx
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx
        elif self.aggrmethod == "nores":
            return x

    def forward(self, input, adj):
        x = input
        denseout = None
        # Here out is the result in all levels.
        for gc in self.hiddenlayers:
            denseout = self._doconcat(denseout, x)
            x = gc(x, adj)
            x = F.dropout(x, self.dropout, training=self.training)

        if not self.dense:
            return self._doconcat(x, input)
        return self._doconcat(x, denseout)

    def get_outdim(self):
        return self.out_features

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.in_features,
                                              self.hiddendim,
                                              self.nhiddenlayer,
                                              self.out_features)

class DenseGCNBlock(Module):
    """
    The multiple layer GCN with dense connection block.
    """

    def __init__(self, in_features, out_features, nbaselayer, T2, group2, weight_beta, Ortho,
                 withbn=True, withloop=True, activation=F.relu, dropout=True,
                 aggrmethod="concat", dense=True, ):

        super(DenseGCNBlock, self).__init__()
        self.model = GraphBaseBlock(in_features=in_features,
                                    out_features=out_features,
                                    nbaselayer=nbaselayer,
                                    withbn=withbn,
                                    withloop=withloop,
                                    activation=activation,
                                    dropout=dropout,
                                    dense=True,
                                    aggrmethod=aggrmethod,
                                    T2=T2,
                                    group2=group2,
                                    weight_beta=weight_beta,
                                    Ortho=Ortho
                                    )

    def forward(self, input, adj):
        return self.model.forward(input, adj)

    def get_outdim(self):
        return self.model.get_outdim()

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (self.__class__.__name__,
                                              self.aggrmethod,
                                              self.model.in_features,
                                              self.model.hiddendim,
                                              self.model.nhiddenlayer,
                                              self.model.out_features)

