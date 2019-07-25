import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d, LayerNorm, Parameter
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


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = Linear(d_model, d_model)
        self.v_linear = Linear(d_model, d_model)
        self.k_linear = Linear(d_model, d_model)
        self.dropout = Dropout(dropout)
        self.out = Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

def attn_matrix(A, X, attn_weight):
    # A : [batch, N, N]
    # X : [batch, N, F']
    # weight_attn : F'
    num_atoms = int(X.get_shape()[1])
    hidden_dim = int(X.get_shape()[2])

    _X1 = tf.einsum('ij,ajk->aik', attn_weight, tf.transpose(X, [0,2,1]))
    _X2 = tf.matmul(X, _X1)
    _A = tf.multiply(A, _X2)
    _A = tf.nn.tanh(_A)

    return _A

def graph_attn(adjacency_matrix, node_features, weight, bias, attn):
    dim = int(weight[0].get_shape()[1])
    num_atoms = int(adjacency_matrix.get_shape()[1])

    node_features_total = []
    adjacency_total = []

    for i in range( len(weight) ):
        _b = tf.reshape( tf.tile( bias[i], [num_atoms] ), [num_atoms, dim] )
        _h = tf.einsum('ijk,kl->ijl', node_features, weight[i]) + _b
        _adjacency_matrix = attn_matrix(adjacency_matrix, _h, attn[i])
        _h = tf.nn.relu(tf.matmul(_adjacency_matrix, _h))
        node_features_total.append(_h)
        adjacency_total.append(_adjacency_matrix)

    _node_features = tf.nn.relu(tf.reduce_mean(node_features_total, 0))
    _adjacency_matrix = tf.reduce_mean(adjacency_total, 0)

    _node_features = get_skip_connection(_node_features, node_features)

    return _node_features, _adjacency_matrix


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
