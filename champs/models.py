import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d, LayerNorm
from torch_geometric import nn

from .layers import GatedEdgeConv, GATEdgeConv, NormGRU


class Net(torch.nn.Module):
    """GCN model using a gated NNConv, modified from code by Heng CherKeng."""
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
        self.conv = GATEdgeConv(dim, dim, enc, aggr='mean')
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


class PathNet(torch.nn.Module):
    """Model using GCN atom encodings and a path-based readout."""
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

        # encoding step
        for i in range(self.processing_steps):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m, h)

        # Out is now an atom-level representation. Now we read out using the target paths.
        path_atoms = torch.split(data.target_path, 1, dim=0)
        atoms = [torch.index_select(out, dim=0, index=a.view(-1)) for a in path_atoms]
        # NB this does not explicitly include bond information or any ordering information

        s2s = self.set2set(out, data.batch)  # molecule-level representation
        s2s0 = torch.index_select(s2s, dim=0, index=data.batch)  # one for each atom
        s2s0 = torch.index_select(s2s0, dim=0, index=path_atoms[0].view(-1)) # now one for each edge

        predict = self.predict(torch.cat(atoms + [s2s0], -1))
        predict = torch.gather(predict, 1, data.target_class.view(-1, 1)).squeeze(-1)
        return predict
