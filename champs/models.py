import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d, LayerNorm

from torch_geometric import nn

from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils import scatter_
import math



class Net(torch.nn.Module):
    def __init__(self, num_features, dim, n_outputs=8, edge_dim=4):
        super().__init__()
        self.preprocess = torch.nn.Sequential(
            LinearBn(num_features, 128),
            torch.nn.ReLU(inplace=True),
            LinearBn(128, 128),
            torch.nn.ReLU(inplace=True),
        )

        encoder = torch.nn.Sequential(
            LinearBn(edge_dim, 256),
            torch.nn.ReLU(inplace=True),
            LinearBn(256, 256),
            torch.nn.ReLU(inplace=True),
            LinearBn(256, 128),
            torch.nn.ReLU(inplace=True),
            LinearBn(128, dim * dim),
            #torch.nn.ReLU(inplace=True),
        )

        self.conv = nn.NNConv(dim, dim, encoder, aggr='mean', root_weight=False)
        self.gru = GRU(dim, dim)

        self.set2set = nn.Set2Set(dim, processing_steps=3)
        #predict coupling constant
        self.predict = torch.nn.Sequential(
            LinearBn(4*128, 1024),  #node_hidden_dim
            torch.nn.ReLU(inplace=True),
            LinearBn(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, n_outputs),
        )

    def forward(self, data):
        out = self.preprocess(data.x)
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        # mol_s2s = torch.index_select(self.set2set(out, data.batch), dim=0, index=data.batch)
        # print(out.shape)
        # out = torch.cat([out, mol_s2s], -1)
        # print(out.shape)

        # out is now an atom-level representation
        # now need to run a dense layer over (atom1, atom2) pairs
        atom0, atom1 = torch.split(data.target_index, 1, dim=0)
        node0 = torch.index_select(out, dim=0, index=atom0.view(-1))
        node1 = torch.index_select(out, dim=0, index=atom1.view(-1))

        # add set2set output over atoms
        atom_index = torch.arange(0, out.shape[0], device=out.device).long()

        s2s = self.set2set(out, data.batch)
        s2s = torch.index_select(s2s, dim=0, index=data.batch)

        predict = self.predict(torch.cat([node0, node1, s2s], -1))
        predict = torch.gather(predict, 1, data.target_class.view(-1, 1)).squeeze(-1)
        return predict


class SingleTargetNet(torch.nn.Module):
    def __init__(self, num_features, dim, n_outputs=8):
        super().__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)

        nn = Sequential(
            Linear(4, 128),
            LayerNorm(128),
            ReLU(),
            Linear(128, dim * dim),
            LayerNorm(dim * dim),
        )
        self.conv = NNConv(dim, dim, nn, aggr='mean', root_weight=False)
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = Sequential(
            Linear(6 * dim, dim),
            LayerNorm(dim),
            ReLU(),
        )
        self.lin2 = Linear(dim, n_outputs)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        # mol_s2s = torch.index_select(self.set2set(out, data.batch), dim=0, index=data.batch)
        # print(out.shape)
        # out = torch.cat([out, mol_s2s], -1)
        # print(out.shape)

        # out is now an atom-level representation
        # now need to run a dense layer over (atom1, atom2) pairs
        atom0, atom1 = torch.split(data.target_index, 1, dim=0)
        node0 = torch.index_select(out, dim=0, index=atom0.view(-1))
        node1 = torch.index_select(out, dim=0, index=atom1.view(-1))

        # add set2set output over atoms
        atom_index = torch.arange(0, out.shape[0], device=out.device).long()

        s2s = self.set2set(out, atom_index)
        s2s0 = torch.index_select(s2s, dim=0, index=atom0.view(-1))
        s2s1 = torch.index_select(s2s, dim=0, index=atom1.view(-1))

        predict = F.relu(self.lin1(torch.cat([node0, node1, s2s0, s2s1], -1)))
        predict = self.lin2(predict)
        predict = torch.gather(predict, 1, data.target_class.view(-1, 1)).squeeze(-1)
        return predict


class LinearBn(torch.nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super().__init__()
        self.linear = torch.nn.Linear(in_channel, out_channel, bias=False)
        self.bn   = torch.nn.BatchNorm1d(out_channel,eps=1e-05, momentum=0.1)
        self.act  = act

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x



class Set2Set(torch.torch.nn.Module):

    def softmax(self, x, index, num=None):
        x = x -  scatter_max(x, index, dim=0, dim_size=num)[0][index]
        x = x.exp()
        x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-16)
        return x

    def __init__(self, in_channel, processing_step=1):
        super(Set2Set, self).__init__()
        num_layer = 1
        out_channel = 2 * in_channel

        self.processing_step = processing_step
        self.in_channel  = in_channel
        self.out_channel = out_channel
        self.num_layer   = num_layer
        self.lstm = torch.nn.LSTM(out_channel, in_channel, num_layer)
        self.lstm.reset_parameters()

    def forward(self, x, batch_index):
        batch_size = batch_index.max().item() + 1

        h = (x.new_zeros((self.num_layer, batch_size, self.in_channel)),
             x.new_zeros((self.num_layer, batch_size, self.in_channel)))

        q_star = x.new_zeros(batch_size, self.out_channel)
        for i in range(self.processing_step):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, -1)

            e = (x * q[batch_index]).sum(dim=-1, keepdim=True) #shape = num_node x 1
            a = self.softmax(e, batch_index, num=batch_size)   #shape = num_node x 1
            r = scatter_add(a * x, batch_index, dim=0, dim_size=batch_size) #apply attention #shape = batch_size x ...
            q_star = torch.cat([q, r], dim=-1)

        return q_star


class HengNet(torch.nn.Module):
    def __init__(self, node_dim, dim, num_target=8, edge_dim=4):
        super().__init__()
        self.num_propagate = 6
        self.num_s2s = 6

        self.preprocess = torch.nn.Sequential(
            LinearBn(node_dim, 128),
            torch.nn.ReLU(inplace=True),
            LinearBn(128, 128),
            torch.nn.ReLU(inplace=True),
        )

        encoder = torch.nn.Sequential(
            LinearBn(edge_dim, 256),
            torch.nn.ReLU(inplace=True),
            LinearBn(256, 256),
            torch.nn.ReLU(inplace=True),
            LinearBn(256, 128),
            torch.nn.ReLU(inplace=True),
            LinearBn(128, node_dim * node_dim),
            #torch.nn.ReLU(inplace=True),
        )

        self.conv = nn.NNConv(dim, dim, encoder, aggr='mean', root_weight=False)
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(128, processing_step=self.num_s2s)


        #predict coupling constant
        self.predict = torch.nn.Sequential(
            LinearBn(4*128, 1024),  #node_hidden_dim
            torch.nn.ReLU(inplace=True),
            LinearBn(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, num_target),
        )

    def forward(self, data):
        out = self.preprocess(data.x)
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        # out is now an atom-level representation
        # now need to run a dense layer over (atom1, atom2) pairs
        atom0, atom1 = torch.split(data.target_index, 1, dim=0)
        node0 = torch.index_select(out, dim=0, index=atom0.view(-1))
        node1 = torch.index_select(out, dim=0, index=atom1.view(-1))

        s2s = self.set2set(out, data.batch)
        s2s = torch.index_select(s2s, dim=0, index=data.batch)

        predict = self.predict(torch.cat([node0, node1, s2s], -1))
        predict = torch.gather(predict, 1, data.target_class.view(-1, 1)).squeeze(-1)
        return predict
