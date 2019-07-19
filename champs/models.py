import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d, LayerNorm

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa


class Net(torch.nn.Module):
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

        s2s = self.set2set(out)
        s2s0 = torch.index_select(s2s, dim=0, index=atom0.view(-1))
        s2s1 = torch.index_select(s2s, dim=0, index=atom1.view(-1))

        predict = F.relu(self.lin1(torch.cat([node0, node1, s2s0, s2s1], -1)))
        predict = self.lin2(predict)
        predict = torch.gather(predict, 1, data.target_class.view(-1, 1)).squeeze(-1)
        return predict
