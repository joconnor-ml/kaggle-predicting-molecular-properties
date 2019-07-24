import sys

sys.path.append("..")
from champs.datasets import ChampsDatasetTargetSubset
from champs.models import Net
from champs.training import mae, train_subset, test, test_one
from torch_geometric.data import DataLoader
import torch
import numpy as np



def main(target_classes, initial_checkpoint, model_name):
    torch.manual_seed(0)  # for reproducabibilites

    dim = 64

    dataset = ChampsDatasetTargetSubset("./data/", targets=target_classes)
    # Normalize targets to mean = 0 and std = 1.
    sum = dataset.data.y.sum(dim=0)
    sum2 = (dataset.data.y ** 2).sum(dim=0)
    nonzero = (dataset.data.y != 0).sum(dim=0).float()
    mean = sum / nonzero
    std = (sum2 / nonzero - mean ** 2) ** 0.5

    print(mean, std)
    dataset.data.y = (dataset.data.y - mean) / std

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, dataset[0].edge_attr.shape[-1], dim, processing_steps=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)

    if initial_checkpoint is not None:
        checkpoint = torch.load(initial_checkpoint, map_location=device)
        model.load_state_dict(checkpoint)

        initial_optimizer = initial_checkpoint.replace('_model.pth','_optimizer.pth')
        checkpoint  = torch.load(initial_optimizer)
        start_epoch = checkpoint['epoch']
        last_val_loss = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Read model from {}. Resuming training from epoch {}. Last val loss = {:04d}".format(checkpoint, start_epoch, last_val_loss))

    for epoch in range(start_epoch, 501):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train_subset(model, optimizer, train_loader, device, epoch, target_classes, std)

        # if 0:
        if epoch % 10 == 1:
            val_error = test(model, val_loader, device, std)
            val_errors = [np.log(test_one(model, val_loader, i, device, std))
                          for i in target_classes]

            scheduler.step(val_error)
            print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation score: {:.7f}'.format(epoch, lr, loss, val_error))
            print(", ".join(["target {}: {:.5f}".format(i, val_errors[i]) for i in target_classes]))

            torch.save(model.state_dict(), './checkpoint/{}.{:04d}_model.pth'.format(model_name, epoch))
            torch.save({
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_error,
            }, './checkpoint/{}.{:04d}_optimizer.pth'.format(model_name, epoch))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("--target-classes", "-t", nargs="+", type=int, default=None)
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    args = parser.parse_args()

    main(target_classes=args.target_classes, initial_checkpoint=args.checkpoint,
         model_name=args.model_name)
