# -*- coding: utf-8 -*-
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from torch_geometric.nn.conv import MessagePassing
#import dgl.nn.pytorch
from GCNLayer import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size).apply(init),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False).apply(init)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)


class HANLayer(nn.Module):

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):

        super(HANLayer, self).__init__()
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GraphConv(in_size, out_size, activation=F.relu).apply(init))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                    g, meta_path)
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[0](new_g, h).flatten(1))

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)

class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, dropout, num_heads=1):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.predict = nn.Linear(hidden_size * num_heads, out_size, bias=False).apply(init)
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads, dropout)
        )

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h)


class HAN_DTI(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, dropout):
        super(HAN_DTI, self).__init__()
        self.sum_layers = nn.ModuleList()

        for i in range(0, len(all_meta_paths)):
            self.sum_layers.append(
                HAN(all_meta_paths[i], in_size[i], hidden_size[i], out_size[i], dropout))


    def forward(self, s_g, s_h_1, s_h_2):
        h1  = self.sum_layers[0](s_g[0], s_h_1)
        h2  = self.sum_layers[1](s_g[1], s_h_2)
        return h1, h2


class GCN(nn.Module):
    def __init__(self, nfeat, hidden_size1,hidden_size2, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, hidden_size1)
        self.gc2 = GraphConvolution(hidden_size1, hidden_size2)
        self.dropout = dropout

    def forward(self, x, adj):
        x = x.to(device)
        adj = adj.to(device)
        x1 = F.relu(self.gc1(x, adj), inplace=True)
        x1 = F.dropout(x1, self.dropout)
        x2 = self.gc2(x1, adj)
        res = x2
        return res


class GCN1(nn.Module):
    def __init__(self,in_size, hidden_size, out_size, dropout):
        super(GCN1, self).__init__()
        self.gc1 = GraphConvolution(in_size, hidden_size)
        self.gc2= GraphConvolution(hidden_size, out_size)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x



class MLP(nn.Module):
    def __init__(self, nfeat,):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(nfeat, 32, bias=False).apply(init),
            nn.ELU(),
            nn.Linear(32, 2, bias=False),
            nn.LogSoftmax(dim=1))
            #nn.Sigmoid())
    def forward(self, x):
        output = self.MLP(x)
        return output

class GATLay(torch.nn.Module):
    def __init__(self, in_features, hid_features, out_features, n_heads):
        super(GATLay, self).__init__()
        self.attentions = [GAL(in_features, hid_features) for _ in
                           range(n_heads)]
        self.out_att = GAL(hid_features * n_heads, out_features)

    def forward(self, x, edge_index, dropout):
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, dropout, training=self.training)
        x = F.elu(self.out_att(x, edge_index))
        return F.softmax(x, dim=1)

class GAL(MessagePassing):
    def __init__(self, in_features, out_featrues):
        super(GAL, self).__init__()
        self.a = torch.nn.Parameter(torch.zeros(size=(2 * out_featrues, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化
        # 定义leakyrelu激活函数
        self.leakyrelu = torch.nn.LeakyReLU()
        self.linear = torch.nn.Linear(in_features, out_featrues)

    def forward(self, x, edge_index):
        x = self.linear(x)
        N = x.size()[0]
        row, col = edge_index
        # col = col.long()
        # row = row.long()
        a_input = torch.cat([x[row], x[col]], dim=1)

        temp = torch.mm(a_input, self.a).squeeze()
        e = self.leakyrelu(temp)
        e_all = torch.zeros(N)
        for i in range(len(row)):
            e_all[row[i]] += math.exp(e[i])

        # f = open("atten.txt", "w")

        for i in range(len(e)):
            e[i] = math.exp(e[i]) / e_all[row[i]]
        #     f.write("{:.4f}\t {} \t{}\n".format(e[i], row[i], col[i]))
        #
        # f.close()
        return self.propagate(edge_index, x=x, norm=e)

    def message(self, x_j, norm):

        return norm.view(-1, 1) * x_j

class GAT(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout):
        super(GAT, self).__init__()
        self.gatlay = GATLay(in_size, hidden_size, out_size, 4)
        self.dropout = dropout

    def forward(self, x, adj):
        edge_index = adj
        x = self.gatlay(x, edge_index, self.dropout)
        return x


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta



class DMNDTI(nn.Module):
    def __init__(self, num_drug,num_protein,all_meta_paths, in_size, hidden_size, hidden_size1,out_size, dropout):
        super(DMNDTI, self).__init__()
        self.HAN_DTI = HAN_DTI(all_meta_paths, in_size, hidden_size,out_size, dropout)
        #加入的
        hidden_size2 = int(hidden_size1/2)
        hidden_size3 = int(hidden_size1*2)
        self.GCN = GCN(num_drug,hidden_size3, hidden_size1,dropout)
        self.GCN = GCN(num_protein,hidden_size3, hidden_size1,dropout)
        self.SGAT1 = GAT(hidden_size3, hidden_size1, hidden_size2, dropout)
        self.SGAT2 = GAT(hidden_size3, hidden_size1, hidden_size2, dropout)
        # DAE 100d
        self.CGCN = GCN1(456, hidden_size1, hidden_size2, dropout)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(hidden_size1, 1)))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention1 = Attention(hidden_size1)
        self.attention = Attention(hidden_size2)
        self.tanh = nn.Tanh()
        self.MLP = MLP(hidden_size2)


    def forward(self, graph, h, dateset_index, data, iftrain=True, d=None, p=None):
        if iftrain:
            drug, pro = self.HAN_DTI(graph, h[0], h[1])
            drug = drug.to(torch.float32)
            pro = pro.to(torch.float32)
            d1 = torch.tensor(np.loadtxt("../dtiseed/DAE_model/drug_dae.txt"))
            p1 = torch.tensor(np.loadtxt("../dtiseed/DAE_model/protein_dae.txt"))
            d1 = d1.to(torch.float32)
            p1 = p1.to(torch.float32)
            d =  torch.cat([drug, d1], dim=1)
            p = torch.cat([pro, p1], dim=1)
            d= d
            p = p
        #创建topology和 余弦定义的sesmatic
        edge, as_edge,feature = constructure_graph(data, d,p )
        f_edge, as_fedge,f_feature = constructure_knngraph(data,d, p)

        com1 = self.CGCN(feature, edge) 
        com2 = self.CGCN(f_feature, f_edge)  
        emb1 = self.CGCN(feature, edge) 
        emb2 = self.CGCN(f_feature, f_edge)
        Xcom = (com1 + com2)/2
        emb = torch.stack([emb1, emb2,Xcom], dim=1)
        emb, att = self.attention(emb)

        pred1 = self.MLP(emb[dateset_index])

        if iftrain:
            return d,p,pred1
        return pred1




def init(i):
    if isinstance(i, nn.Linear):
        torch.nn.init.xavier_uniform_(i.weight)
