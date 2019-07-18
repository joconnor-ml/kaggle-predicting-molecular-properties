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
    parser.add_argument('checkpoint_file', type=str)
    args = parser.parse_args()

    target = 0
    dim = 64

    test_dataset = ChampsTestDataset("./data/")
    # Normalize targets to mean = 0 and std = 1.
    mean = 15.9187472
    std = 34.93184728

    # Split datasets.
    test_loader = DataLoader(
        test_dataset, batch_size=64,
        num_workers=2,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(test_dataset.num_features, dim).to(device)
    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint)

    def test(loader):
        model.eval()
        error = 0

        preds = []
        for data in loader:
            data = data.to(device)
            preds.append(model(data).detach().cpu().numpy())

        return np.concatenate(preds)

    preds = test(test_loader)
    preds = pd.Series(preds) * std + mean
    print(preds.head())
    preds.to_csv("output/preds.test.csv")
