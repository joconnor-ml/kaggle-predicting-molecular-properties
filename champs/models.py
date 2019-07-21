import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d, LayerNorm
from torch_geometric import nn



import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
import math


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


class GatedEdgeConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr='add',
                 bias=True,
                 **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr
        self.lin0 = Linear(self.out_channels, 32, bias=False)
        self.lin1 = Linear(self.out_channels, 32, bias=True)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        uniform(self.in_channels, self.root)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def get_gate_coeff(self, x0, x1):
        x0 = self.lin0(x0)
        x1 = self.lin1(x1)
        return F.sigmoid(x0 + x1)

    def message(self, x_j0, pseudo):
        print(x_j0.shape)
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        x_j1 = torch.matmul(x_j0.unsqueeze(1), weight).squeeze(1)
        print(x_j1.shape)
        # add gated skip:
        coeff = self.get_gate_coeff(x_j0, x_j1)
        out = x_j0 * coeff + x_j1 * (1.0 - coeff)
        print(out.shape)
        return out


    def update(self, aggr_out, x):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class NormGRU(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gru = GRU(in_dim, out_dim)
        self.norm = LayerNorm(out_dim)

    def forward(self, m, h):
        out, h = self.gru(m.unsqueeze(0), h)
        out = out.squeeze(0)
        out = self.norm(out)
        return out, h


class Net(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, dim, n_outputs=8, processing_steps=3):
        super().__init__()

        self.processing_steps = processing_steps

        self.preprocess = Sequential(
            Linear(num_node_features, dim),
            BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim),
            BatchNorm1d(dim),
            ReLU(),
        )

        enc = Sequential(
            Linear(num_edge_features, dim),
            BatchNorm1d(dim),
            ReLU(),
            Linear(dim, dim * dim),
            BatchNorm1d(dim * dim),
        )
        self.conv = GatedEdgeConv(dim, dim, enc, aggr='mean')
        self.gru = NormGRU(dim, dim)

        self.set2set = nn.Set2Set(dim, processing_steps=self.processing_steps)
        self.predict = Sequential(
            Linear(4 * dim, 4 * dim),
            LayerNorm(4 * dim),
            ReLU(),
            Linear(4 * dim, 4 * dim),
            LayerNorm(4 * dim),
            ReLU(),
            Linear(4 * dim, n_outputs),
        )

    def forward(self, data):
        out = self.preprocess(data.x)
        h = out.unsqueeze(0)

        for i in range(self.processing_steps):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m, h)

        # mol_s2s = torch.index_select(self.set2set(out, data.batch), dim=0, index=data.batch)
        # print(out.shape)
        # out = torch.cat([out, mol_s2s], -1)
        # print(out.shape)

        # out is now an atom-level representation
        # now need to run a dense layer over (atom1, atom2) pairs
        atom0, atom1 = torch.split(data.target_index, 1, dim=0)
        node0 = torch.index_select(out, dim=0, index=atom0.view(-1))
        node1 = torch.index_select(out, dim=0, index=atom1.view(-1))

        s2s = self.set2set(out, data.batch)  # molecule-level representation
        s2s0 = torch.index_select(s2s, dim=0, index=data.batch)  # one for each atom
        s2s0 = torch.index_select(s2s0, dim=0, index=atom0.view(-1)) # now one for each edge

        predict = self.predict(torch.cat([node0, node1, s2s0], -1))
        predict = torch.gather(predict, 1, data.target_class.view(-1, 1)).squeeze(-1)
        return predict
