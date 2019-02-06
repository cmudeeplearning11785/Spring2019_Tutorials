# coding=utf-8

import config

import torch
import torch.nn as nn
import numpy as np

device = "cuda" if config.cuda else "cpu"


def run_one_epoch(model, loader, criterion, update=True, optimizer=None):
    cum_loss = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = criterion(output, y)
        if update:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        cum_loss += loss.cpu().detach().numpy().item()
    return cum_loss/len(loader)


def train(model, train_dataloader, val_dataloader):
    max_epoch = config.max_epoch
    if config.sanity:
        max_epoch = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    model = model.to(device)
    best_loss = float("+inf")

    print('begin training')
    lr = config.lr
    epoch = 0
    pat = 0
    while True:
        epoch += 1
        print("Epoch", epoch)
        model.train()
        train_loss = run_one_epoch(model, train_dataloader, criterion, True, optimizer)
        print("Training loss :", train_loss)
        model.eval()
        val_loss = run_one_epoch(model, val_dataloader, criterion, False)
        print("Validation loss :", val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            model.save()
            pat = 0
        else:
            pat += 1
            if pat >= config.patience:
                print("Early stopping !")
                break
        if epoch >= max_epoch:
            print('reached maximum number of epochs !')
            break


def predict(model, loader):
    l = []
    for x, in loader:
        x = x.to(device)
        output = model(x)
        classes = np.argmax(output.cpu().detach().numpy(), axis=1)
        l.append(classes)
    return np.concatenate(l)
