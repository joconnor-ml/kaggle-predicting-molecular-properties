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
    test_dataset = dataset[::5]
    test_loader = DataLoader(
        test_dataset, batch_size=64,
        num_workers=2,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(test_dataset.num_features, dim).to(device)
    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint)

    # Normalize targets to mean = 0 and std = 1.
    mean = torch.from_numpy(
        np.array([14.4633,  0.4420, -0.0663, -0.8348,  0.0800,  1.1960,  0.6049,  0.0354]),
    ).to(device)
    std = torch.from_numpy(
        np.array([34.8631,  4.6798,  2.2415,  3.0292,  0.7675,  2.4574,  2.0641,  0.3092])
    ).to(device)

    def test(loader):
        model.eval()
        error = 0

        preds = []
        for data in loader:
            data = data.to(device)
            p = model(data) * std + mean
            p = torch.gather(p, 1, data.target_class.view(-1, 1)).squeeze(-1)
            preds.append(p.detach().cpu().numpy())

        return np.concatenate(preds)

    preds = test(test_loader)
    preds = pd.Series(preds)
    print(preds.head())
    preds.to_csv("output/preds.test.csv")
