import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter


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
class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.bottom_gcn = GCN(dim, dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(Ortho_GCN(dim, dim, act, drop_p))#Ortho_GCN
            self.up_gcns.append(Ortho_GCN(dim, dim, act, drop_p))#Ortho_GCN
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

    def forward(self, g, h):
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = h
        for i in range(self.l_n):
            h = self.down_gcns[i](g, h)
            adj_ms.append(g)
            down_outs.append(h)
            g, h, idx = self.pools[i](g, h)
            indices_list.append(idx)
        h = self.bottom_gcn(g, h)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx)
            h = self.up_gcns[i](g, h)
            h = h.add(down_outs[up_idx])
            hs.append(h)
        h = h.add(org_h)
        hs.append(h)
        return hs


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        h = self.drop(h)
        h = torch.matmul(g, h)
        h = self.proj(h)
        h = self.act(h)
        return h

class Ortho_GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(Ortho_GCN, self).__init__()
        # self.proj = nn.Linear(in_dim, out_dim)
        self.in_features = in_dim
        self.out_features = out_dim
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        nn.init.xavier_uniform_(self.weight.data)
        self.weight_normalization = Ortho_Trans(T=4, norm_groups=2)
        self.weight_beta = 1
        self.t = None
        self.we = None
    # def reset_parameters(self):
    #     stdv = 1. / np.math.sqrt(self.out_features)
    #     self.weight.data.uniform_(-stdv, stdv)
    def forward(self, g, h):

        h = self.drop(h)
        h = torch.matmul(g, h)
        I = torch.eye(self.in_features).cuda()
        self.we = self.weight_beta * self.weight.cuda() + (1 - self.weight_beta) * I
        self.t = self.weight_normalization(self.we)
        if self.training:
            self.t.retain_grad()
        output = torch.mm(h, self.t)
        # h = self.proj(h)
        h = self.act(output)
        return h

class GraphConvolution_Ortho(nn.Module):

    def __init__(self, in_features, out_features,T2, group2,weight_beta, Ortho):
        super(GraphConvolution_Ortho, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.Ortho = Ortho
        self.weight_beta = weight_beta
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.weight_normalization = Ortho_Trans(T=T2, norm_groups=group2)#2，2 2,1            cora(5141) （42  44）
        self.reset_parameters()
        self.t = None
        self.we = None
        # self.W = Parameter(torch.FloatTensor(self.in_features, self.out_features))

    def reset_parameters(self):
        stdv = 1. / np.math.sqrt(self.out_features)
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
            # output = self.weight_beta * torch.mm(hi, self.t) + (1 - self.weight_beta) * torch.mm(hi, I)

        else:
            output = torch.mm(hi, self.weight)
        # I = torch.eye(self.in_features).cuda()
        # self.we = self.weight_beta * self.weight.cuda() + (1 - self.weight_beta) * I
        # output = torch.mm(hi, self.we)
        return output

class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g


class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)
