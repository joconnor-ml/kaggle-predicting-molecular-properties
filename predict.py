import sys
sys.path.append("..")
from champs.datasets import ChampsDatasetMultiTarget
from champs.models import Net
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch

import pandas as pd
import numpy as np

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_file', type=str)
    parser.add_argument('outfile', type=str)
    args = parser.parse_args()

    target = 0
    dim = 64

    dataset = ChampsDatasetMultiTarget("./data/")

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 10000, 10000])

    train_loader = DataLoader(
        train_dataset, batch_size=64,
        num_workers=2,
        pin_memory=True,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64,
        num_workers=2,
        pin_memory=True,
        shuffle=True
    )

    device = torch.device(args.device)
    model = Net(dataset.num_features, dataset[0].edge_attr.shape[-1], dim).to(device)
    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint)

    def test(loader):
        model.eval()

        preds = []
        for data in loader:
            data = data.to(device)
            preds.append(model(data).detach().cpu().numpy())

        return np.concatenate(preds)

    preds = test(val_loader)
    preds = pd.Series(preds)
    print(preds.head())
    preds.to_csv(args.outfile)
