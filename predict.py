import sys
sys.path.append("..")
from champs.datasets import ChampsDataset
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

    dataset = ChampsDataset("./data/")

    # Split datasets.
    test_dataset = dataset[::2]
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

        preds = []
        for data in loader:
            data = data.to(device)
            preds.append(model(data).detach().cpu().numpy())

        return np.concatenate(preds)

    preds = test(test_loader)
    preds = pd.Series(preds)
    print(preds.head())
    preds.to_csv(args.outfile)
