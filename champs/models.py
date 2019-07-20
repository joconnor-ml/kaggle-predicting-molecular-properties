import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, BatchNorm1d, LayerNorm
from torch_geometric import nn


class Net(torch.nn.Module):
    def __init__(self, num_features, dim, n_outputs=8):
        super().__init__()
        self.preprocess = Sequential(
            Linear(num_features, dim),
            LayerNorm(dim),
            ReLU(),
            Linear(dim, dim),
            LayerNorm(dim),
            ReLU(),
        )

        enc = Sequential(
            Linear(4, dim),
            LayerNorm(dim),
            ReLU(),
            Linear(dim, dim * dim),
            LayerNorm(dim * dim),
        )
        self.conv = nn.NNConv(dim, dim, enc, aggr='mean', root_weight=False)
        self.gru = Sequential(
            GRU(dim, dim),
            LayerNorm(dim),
        )

        self.set2set = nn.Set2Set(dim, processing_steps=3)
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

        s2s = self.set2set(out, data.target_batch_index)
        s2s0 = torch.index_select(s2s, dim=0, index=data.target_batch_index)

        predict = self.predict(torch.cat([node0, node1, s2s0], -1))
        predict = torch.gather(predict, 1, data.target_class.view(-1, 1)).squeeze(-1)
        return predict
