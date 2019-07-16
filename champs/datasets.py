import os
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar,
                                  Data, Dataset)


class ChampsDatasetBasic(Dataset):
    num_classes = 1

    def __init__(self, csv_file, graph_dir):
        self.csv_file = pd.read_csv(csv_file)
        self.graph_dir = graph_dir
        self.id = list(self.csv_file["molecule_name"].unique())

    def __len__(self):
        return (len(self.id))

    def __getitem__(self, index):
        molecule_name = self.id[index]
        edge_array = np.load(os.path.join(self.graph_dir, "{}.edge_array.npy".format(molecule_name)))
        edge_features = np.load(os.path.join(self.graph_dir, "{}.edge_features.npy".format(molecule_name)))
        atom_features = np.load(os.path.join(self.graph_dir, "{}.atom_features.npy".format(molecule_name)))
        targets = np.load(os.path.join(self.graph_dir, "{}.targets.npy".format(molecule_name)))
        target_indices = np.load(os.path.join(self.graph_dir, "{}.target_indices.npy".format(molecule_name)))

        return Data(
            x=torch.from_numpy(atom_features).type(torch.FloatTensor),
            edge_index=torch.from_numpy(edge_array),
            edge_attr=torch.from_numpy(edge_features).type(torch.FloatTensor),
            y=torch.from_numpy(targets.reshape(-1, 1)).type(torch.FloatTensor),
            target_indices=torch.from_numpy(target_indices),
        )


class ChampsDataset(InMemoryDataset):
    num_classes = 1
    csv_file = "/mnt/kaggle-predicting-molecular-properties/data/csv/train.csv"
    graph_dir = "/mnt/kaggle-predicting-molecular-properties/data/graphs/"

    def __init__(self, root, transform=None, pre_transform=None):
        super(ChampsDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['champs.dataset']

    def download(self):
        pass

    def process(self):
        csv_file = pd.read_csv(self.csv_file)

        id = list(csv_file["molecule_name"].unique())

        data_list = []
        for molecule_name in id:
            edge_array = np.load(os.path.join(self.graph_dir, "{}.edge_array.npy".format(molecule_name)))
            edge_features = np.load(os.path.join(self.graph_dir, "{}.edge_features.npy".format(molecule_name)))
            atom_features = np.load(os.path.join(self.graph_dir, "{}.atom_features.npy".format(molecule_name)))
            targets = np.load(os.path.join(self.graph_dir, "{}.targets.npy".format(molecule_name)))
            target_indices = np.load(os.path.join(self.graph_dir, "{}.target_indices.npy".format(molecule_name)))
            target_classes = np.load(os.path.join(self.graph_dir, "{}.target_indices.npy".format(molecule_name)))

            row = Data(
                x=torch.from_numpy(atom_features).type(torch.FloatTensor),
                edge_index=torch.from_numpy(edge_array).type(torch.int64),
                edge_attr=torch.from_numpy(edge_features).type(torch.FloatTensor),
                y=torch.from_numpy(targets.reshape(-1, 1)).type(torch.FloatTensor),
                target_index=torch.from_numpy(target_indices).type(torch.int64),
                target_class=torch.from_numpy(target_classes).type(torch.uint8),
                num_nodes=atom_features.shape[0]
            )

            data_list.append(
                row
            )

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
