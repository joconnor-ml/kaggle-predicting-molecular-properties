import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d, LayerNorm, Parameter, MultiheadAttention
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
        self.lin0 = Linear(self.out_channels, self.out_channels, bias=False)
        self.lin1 = Linear(self.out_channels, self.out_channels, bias=True)

        self.register_parameter('root', None)

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

    def gated_skip_connection(self, x0, x1):
        coeff = self.get_gate_coeff(x0, x1)
        return x0 * coeff + x1 * (1.0 - coeff)


    def message(self, x_j, pseudo):
        # from NNConv:
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        x_j1 = torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

        # what kind of attention makes sense here?
        # we are updating the representation of atom j with contributions from its neighbours.

        # now add gated skip connection:
        return self.gated_skip_connection(x_j, x_j1)


    def update(self, aggr_out, x):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GATEdgeConv(MessagePassing):
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
        self.lin0 = Linear(self.out_channels, self.out_channels, bias=False)
        self.lin1 = Linear(self.out_channels, self.out_channels, bias=True)
        self.attn = MultiheadAttention(self.out_channels, 4)

        self.register_parameter('root', None)

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

    def gated_skip_connection(self, x0, x1):
        coeff = self.get_gate_coeff(x0, x1)
        return x0 * coeff + x1 * (1.0 - coeff)

    def message(self, x_j, pseudo):
        # from NNConv:
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        x_j1 = torch.matmul(x_j.unsqueeze(1), weight)
        x_j1 = self.attn(x_j1, x_j1, x_j1).squeeze(1)  # self-attention

        # now add gated skip connection:
        return self.gated_skip_connection(x_j, x_j1)


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
