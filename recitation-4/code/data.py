import paths
import config

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_loader(mode="train"):
    if mode == "train":
        data_path = paths.train_data
        labels_path = paths.train_labels
        shuffle = True
    if mode == "val":
        data_path = paths.valid_data
        labels_path = paths.valid_labels
        shuffle = False
    if mode == "test":
        data_path = paths.test_data
        labels_path = None
        shuffle = False
    data = np.load(data_path)
    if config.sanity:
        data = data[:150]

    if labels_path:
        labels = np.load(labels_path)
        if config.sanity:
            labels = labels[:150]

        print(data.shape, labels.shape)
        dataset = TensorDataset(torch.tensor(data, dtype=torch.float),
                                torch.tensor(labels, dtype=torch.long))
    else:
        dataset = TensorDataset(torch.tensor(data, dtype=torch.float))
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=config.batch_size, drop_last=False)

    return dataloader


def get_test_labels():
    return np.load(paths.test_labels)
