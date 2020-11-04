import torch
import torchvision

from util.data_processing import zPad_or_Rescale


def get_MNIST_dataloader(path='/mnt/Linux_Storage/', batch_size=128):
    train_dataset = torchvision.datasets.MNIST(path, train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize(32),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))

    train_dataset = torch.utils.data.random_split(train_dataset, [20000, len(train_dataset)-20000])[0]


    test_dataset = torchvision.datasets.MNIST(path, train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize(32),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ]))

    test_dataset = torch.utils.data.random_split(test_dataset, [5000, len(test_dataset) - 5000])[0]


    train_loader = torch.utils.data.DataLoader(
        train_dataset
        ,
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset
        ,
        batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
