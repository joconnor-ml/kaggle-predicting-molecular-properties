import sys
sys.path.append("..")
from champs.datasets import ChampsTestDataset
from champs.models import Net
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch

import pandas as pd
import numpy as np

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_file", type=str)
    parser.add_argument("--device", "-d", type=str)
    args = parser.parse_args()

    target = 0
    dim = 64

    test_dataset = ChampsTestDataset("./data/")

    # Split datasets.
    test_loader = DataLoader(
        test_dataset, batch_size=64,
        num_workers=2,
        pin_memory=True
    )

    device = torch.device(args.device)
    model = Net(test_dataset.num_features, test_dataset[0].edge_attr.shape[-1], dim).to(device)
    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint)

    def test(loader):
        model.eval()

        preds = []
        for data in loader:
            data = data.to(device)
            preds.append(model(data).detach().cpu().numpy())

        return np.concatenate(preds)

    preds = test(test_loader)
    preds = pd.Series(preds)
    print(preds.head())
    preds.to_csv("output/preds.test.csv")
