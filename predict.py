import sys
sys.path.append("..")
from champs.datasets import ChampsDataset
from champs.models import Net
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_file', type=str)
    args = parser.parse_args()

    target = 0
    dim = 64

    dataset = ChampsDataset("./data/")
    # Normalize targets to mean = 0 and std = 1.
    mean = dataset.data.y[:, target].mean().item()
    std = dataset.data.y[:, target].std().item()
    dataset.data.y[:, target] = (dataset.data.y[:, target] - mean) / std

    # Split datasets.
    val_dataset = dataset[::5]
    val_loader = DataLoader(
        val_dataset, batch_size=256,
        num_workers=2,
        pin_memory=True,
    )

    device = torch.device('cpu')
    model = Net(dataset.num_features, dim).to(device)
    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    print(checkpoint)
    model.load_state_dict(checkpoint)

    def mae(predict, truth):
        predict = predict.view(-1)
        truth = truth.view(-1)

        score = torch.abs(predict-truth)
        score = score.sum()
        return score


    def test(loader):
        model.eval()
        error = 0

        for data in loader:
            data = data.to(device)
            error += mae(model(data), data.y).item()
        return error / len(loader.dataset)

    test(val_loader)