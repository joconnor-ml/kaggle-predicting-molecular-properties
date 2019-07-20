import sys
sys.path.append("..")
from champs.datasets import ChampsSampleDatasetMultiTarget
from champs.models import Net
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch


dim = 32

dataset = ChampsSampleDatasetMultiTarget("./data/")
# Normalize targets to mean = 0 and std = 1.
sum = dataset.data.y.sum(dim=0)
sum2 = (dataset.data.y ** 2).sum(dim=0)
nonzero = (dataset.data.y != 0).sum(dim=0).float()
mean = sum / nonzero
std = (sum2/nonzero - mean**2)**0.5


print(mean, std)
dataset.data.y = (dataset.data.y - mean) / std

# Split datasets.
val_dataset = dataset
train_dataset = dataset
train_loader = DataLoader(
    train_dataset, batch_size=8,
    num_workers=2,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset, batch_size=8,
    num_workers=2,
    pin_memory=True,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, dataset[0].edge_attr.shape[-1], dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=5, min_lr=0.00001)


def log_mae(predict, truth):
    predict = predict.view(-1)
    truth = truth.view(-1)

    score = torch.abs(predict-truth)
    score = score.mean()
    score = torch.log(score)
    return score


def mae(predict, truth, target_class):
    y = torch.gather(truth, 1, target_class.view(-1, 1)).squeeze(-1)
    predict = predict.view(-1)
    y = y.view(-1)

    score = torch.abs(predict-y)
    score = score.sum()
    return score


def weighted_log_mae(predict, truth, weights):
    predict = predict.view(-1)
    truth = truth.view(-1)

    score = torch.abs(predict-truth)
    score = torch.log(score) * weights
    score = score.sum()
    return score


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = mae(model(data), data.y, data.target_class)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += mae(model(data), data.y, data.target_class).item()
    return error / len(loader.dataset)


best_val_error = None
for epoch in range(1, 501):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_error = test(val_loader)
    scheduler.step(val_error)
    print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}'.format(epoch, lr, loss, val_error))

    # if 0:
    if epoch % 10 == 0:
        torch.save(model.state_dict(), './checkpoint/multiscale2.{:04d}_model.pth'.format(epoch))
        torch.save({
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_error,
        }, './checkpoint/sample.{:04d}_optimizer.pth'.format(epoch))

print(mean, std)
