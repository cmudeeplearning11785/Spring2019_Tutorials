import config
import paths

import torch
import torch.nn as nn


class FFNN(nn.Sequential):

    def __init__(self):
        l = []
        sizes = [config.input_size] + config.hidden + [config.nclasses]
        for i in range(len(sizes)-2):
            l.append(nn.Linear(sizes[i], sizes[i+1]))
            if config.dropout > 0:
                l.append(nn.Dropout(config.dropout))
            l.append(nn.ReLU())
        l.append(nn.Linear(sizes[-2], sizes[-1]))
        super(FFNN, self).__init__(*l)

    @staticmethod
    def load():
        model = FFNN()
        path = paths.model
        model.load_state_dict(torch.load(path))

        return model

    def save(self):
        path = paths.model
        torch.save(self.state_dict(), path)
