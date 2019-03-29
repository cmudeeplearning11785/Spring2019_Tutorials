import os
import numpy as np
import torch
import torch.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from inferno.trainers.basic import Trainer
import matplotlib.pyplot as plt


class MNIST(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.mnist = datasets.MNIST(
            root='./data', train=True,
            download=True, transform=transforms.ToTensor())
        self.labeled = True

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        (img, labels) = self.mnist[idx]
        img = img.view(-1)
        if self.labeled:
            return (img, labels)
        else:
            return (img, img)


class LossPrinter(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, *args, **kwargs):
        loss = self.criterion(*args, **kwargs)
        print("Loss: %f" % loss)
        return loss


def train(net, dataset, criterion, num_epochs,
          batch_size, learn_rate, dir_name):
    dir_name = os.path.join('net/', dir_name)
    trainer = Trainer(net[0])

    if (os.path.exists(os.path.join(dir_name, 'model.pytorch'))):
        net_temp = trainer.load_model(dir_name).model
        net[0].load_state_dict(net_temp.state_dict())
        print("Loaded checkpoint directly")
    else:
        if (not os.path.exists(dir_name)):
            os.makedirs(dir_name)
        data_loader = torch.utils.data.DataLoader(
            dataset, shuffle=True, batch_size=batch_size)
        net[0].train()

        trainer \
            .build_criterion(LossPrinter(criterion)) \
            .bind_loader('train', data_loader) \
            .build_optimizer('Adam', lr=learn_rate) \
            .set_max_num_epochs(num_epochs)

        if torch.cuda.is_available():
            trainer.cuda()

        trainer.fit()
        trainer.save_model(dir_name)
    net[0].cpu()
    net[0].eval()


def display_image(arr):
    width = int(np.sqrt(arr.size()[0]))
    arr = arr.cpu().view(width, -1).numpy()
    plt.figure()
    plt.imshow(1.0 - arr, cmap='gray')


def display_reconstruction(net, dataset):
    (image, _) = dataset[np.random.randint(len(dataset))]
    display_image(image)
    image = torch.autograd.Variable(image).unsqueeze(dim=0)
    reconst = net.decode(net.encode(image)).data[0]
    display_image(reconst)


def display_encodings(net, dataset, limits):
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=1000)
    (images, labels) = next(iter(data_loader))
    images = torch.autograd.Variable(images)
    labels = labels.numpy()

    encoded = net.encode(images).data.cpu().numpy()
    encoded = np.sign(encoded) * np.abs(encoded) ** (1.0 / 3.0)
    plt.figure(figsize=(10, 10))
    plt.scatter(encoded[:, 0], encoded[:, 1], c=labels)
    plt.xlim(limits)
    plt.ylim(limits)
    plt.colorbar()


def display_encoding_variation(net, dataset, limits):
    (image, _) = dataset[np.random.randint(len(dataset))]
    display_image(image)

    images = torch.autograd.Variable(image).clone().unsqueeze(dim=0)
    images = images.repeat(20, 1)
    encoded = net.encode(images).data.cpu().numpy()
    encoded = np.sign(encoded) * np.abs(encoded) ** (1.0 / 3.0)
    plt.figure(figsize=(10, 10))
    plt.scatter(encoded[:, 0], encoded[:, 1])
    plt.xlim(limits)
    plt.ylim(limits)


def display_decoding(net, dataset, point):
    point = point ** 3
    point = torch.autograd.Variable(point).unsqueeze(dim=0)
    decoded = net.decode(point).data[0]
    display_image(decoded)
