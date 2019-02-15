import argparse
import sys

import torch
import torch.nn as nn
from inferno.extensions.layers.reshape import Flatten
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from torchvision import datasets, transforms


class MNISTCNNModel(nn.Module):
    def __init__(self):
        super(MNISTCNNModel, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=32, padding=1, kernel_size=3),  # 28*28
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 14*14
            nn.Conv2d(in_channels=32, out_channels=64, padding=1, kernel_size=3),  # 14*14
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7*7
            Flatten(),
            nn.Linear(in_features=64 * 7 * 7, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10)
        ])
        self.firstrun = True

    def forward(self, input):
        h = input
        if self.firstrun:
            print("****************************************")
            print("input: {}".format(h.size()))
        for layer in self.layers:
            h = layer(h)
            if self.firstrun:
                print("{}: {}".format(layer, h.size()))
        if self.firstrun:
            print("****************************************")
        self.firstrun = False
        return h


def mnist_data_loaders(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


def train_model(args):
    model = MNISTCNNModel()
    train_loader, validate_loader = mnist_data_loaders(args)

    # Build trainer
    trainer = Trainer(model) \
        .build_criterion('CrossEntropyLoss') \
        .build_metric('CategoricalError') \
        .build_optimizer('Adam') \
        .validate_every((2, 'epochs')) \
        .save_every((5, 'epochs')) \
        .save_to_directory(args.save_directory) \
        .set_max_num_epochs(args.epochs) \
        .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                        log_images_every='never'),
                      log_directory=args.save_directory)

    # Bind loaders
    trainer \
        .bind_loader('train', train_loader) \
        .bind_loader('validate', validate_loader)

    if args.cuda:
        trainer.cuda()

    # Go!
    trainer.fit()


def main(argv):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--save-directory', type=str, default='output/inferno',
                        help='output directory')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    train_model(args)


if __name__ == '__main__':
    main(sys.argv[1:])
