import sys
sys.path.append("..")
from champs.datasets import ChampsDataset
from champs.models import Net
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch


target = 0
dim = 64

dataset = ChampsDataset("./data/")
# Normalize targets to mean = 0 and std = 1.
mean = dataset.data.y[:, target].mean().item()
std = dataset.data.y[:, target].std().item()
dataset.data.y[:, target] = (dataset.data.y[:, target] - mean) / std

# Split datasets.
val_dataset = dataset[::10]
train_dataset = dataset[1::10]
train_loader = DataLoader(
    train_dataset, batch_size=32,
    num_workers=2,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=32,
    num_workers=2,
    pin_memory=True,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y.squeeze(-1))
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += (model(data) * std - data.y.squeeze(-1) * std).abs().sum().item()  # MAE
    return error / len(loader.dataset)


best_val_error = None
for epoch in range(1, 20):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_error = test(val_loader)
    scheduler.step(val_error)
    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}'.format(epoch, lr, loss, val_error))
