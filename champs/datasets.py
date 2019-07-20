import os
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch_geometric.data import (InMemoryDataset, download_url, extract_tar,
                                  Data, Dataset)

class ChampsDatasetMultiTarget(InMemoryDataset):
    num_classes = 1
    csv_file = "/mnt/kaggle-predicting-molecular-properties/data/csv/train.csv"
    graph_dir = "/mnt/kaggle-predicting-molecular-properties/data/graphs/"

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['champs.multitarget.dataset']

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
            target_classes = np.load(os.path.join(self.graph_dir, "{}.target_class.npy".format(molecule_name)))
            target_weights = np.load(os.path.join(self.graph_dir, "{}.target_weight.npy".format(molecule_name)))

            target_matrix = np.zeros((targets.shape[0], 8), dtype=np.float32)
            target_matrix[np.arange(target_classes.shape[0]), target_classes] = targets

            dx = edge_features[: -3]
            dy = edge_features[: -3]
            dz = edge_features[: -3]

            print(dx.shape, dy.shape, dz.shape)
            r = (dx**2 + dy**2 + dz**2)**0.5
            phi = np.arccos(dz / r)
            theta = np.arctan2(dx, dy)
            edge_features = np.concatenate([edge_features,
                                            phi.reshape(-1, 1),
                                            theta.reshape(-1, 1)], axis=-1)

            row = Data(
                x=torch.from_numpy(atom_features).float(),
                edge_index=torch.from_numpy(edge_array).long(),
                edge_attr=torch.from_numpy(edge_features).float(),
                y=torch.from_numpy(target_matrix).float(),
                target_index=torch.from_numpy(target_indices).long(),
                target_batch_index=torch.from_numpy(np.zeros_like(targets)).long(),
                target_class=torch.from_numpy(target_classes).long(),
                target_weight=torch.from_numpy(target_weights).float(),
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


class ChampsSampleDatasetMultiTarget(ChampsDatasetMultiTarget):
    num_classes = 1
    csv_file = "/mnt/kaggle-predicting-molecular-properties/sample_data/csv/train.csv"
    graph_dir = "/mnt/kaggle-predicting-molecular-properties/sample_data/graphs/"
    @property
    def processed_file_names(self):
        return ['champs.sample.dataset']


class ChampsTestDataset(ChampsDatasetMultiTarget):
    num_classes = 1
    csv_file = "/mnt/kaggle-predicting-molecular-properties/data/csv/test.csv"
    graph_dir = "/mnt/kaggle-predicting-molecular-properties/data/graphs/test"

    @property
    def processed_file_names(self):
        return ['champs.test.dataset']
