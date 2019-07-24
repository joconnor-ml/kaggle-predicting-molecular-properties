"""
Helper functions for training and testing models
"""

import torch


def mae(predict, truth, target_class, std, eval_class=None):
    y = torch.gather(truth, 1, target_class.view(-1, 1)).squeeze(-1)
    predict = predict.view(-1)
    y = y.view(-1)

    score = torch.abs(predict - y)
    if eval_class is not None:
        score = score[target_class == eval_class] * std[eval_class]
    score = score.mean()
    return score


def train_subset(model, optimizer, loader, device, epoch, target_classes, std):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        preds = model(data)
        loss = mae(preds, data.y, data.target_class, std, target_classes[0])
        for i in target_classes[1:]:
            loss += mae(preds, data.y, data.target_class, i)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(loader)  # divide by number of batches


def test(model, loader, device, std):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += mae(model(data), data.y, data.target_class, std).item()
    return error / len(loader)  # divide by number of batches


def test_one(model, loader, eval_class, device, std):
    """Returns mae for a single target class"""
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += mae(model(data), data.y, data.target_class, std, eval_class=eval_class).item()
    return error / len(loader)  # divide by number of batches
