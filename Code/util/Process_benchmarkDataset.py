import torchvision
import os
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from util.data_processing import zPad_or_Rescale, Double_to_Float, imshow_tensor


class DisentangledSpritesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dir, transform=None, train=True):
        """
        Args:
            dir (string): Directory containing the dSprites dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dir = dir
        self.filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.filepath = f'{self.dir}/{self.filename}'
        self.train = train

        dataset_zip = np.load(self.filepath, allow_pickle=True, encoding='bytes')

        GT_class = dataset_zip['latents_classes']
        ### ### conditionally drawn position X and Y in range(13-18)
        posx_mask = [GT_class[i, -2] in range(13, 18) for i in range(GT_class.shape[0])]  # Select only square shape
        posy_mask = [GT_class[i, -1] in range(13, 18) for i in range(GT_class.shape[0])]  # Select only no rotation
        final_mask = [np.all(tup) for tup in zip(posx_mask, posy_mask)]

        self.Unique_index = np.arange(dataset_zip['imgs'].shape[0])

        # print('Keys in the dataset:', dataset_zip.keys())
        self.imgs = dataset_zip['imgs'][final_mask]
        self.Unique_index = self.Unique_index[final_mask]
        # self.latents_values = dataset_zip['latents_values'][final_mask]
        self.latents_classes = GT_class[final_mask]
        self.metadata = dataset_zip['metadata'][()]

        # print('Metadata: \n', self.metadata)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx].astype(np.float32)
        # sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        if self.transform:
            sample = self.transform(sample)

        if self.train:
            return sample, self.latents_classes[idx][1]
        else:
            return sample, self.latents_classes[idx][1], str(self.Unique_index[idx])


# Useful for inference dataset that need to keep track of cell ID
class Desprite_ID_Collator(object):
    """
    Custom collate_fn, specify to dataloader how to batch sample together.
    During inference, it enables to keep track of sample, label as well as single
    cell ids from the file name, and exploit this unique id to retrieve any
    single cell information we want from csv metadata file
    """

    def __call__(self, batch):
        """
        Input is automatically given by the dataloader; a list of size 'batch_size'
        where each element is a tuple as follow : ((sample,file_name),label)
        """
        data = []
        id = []
        label = []
        for i, item in enumerate(batch):
            data.append(item[0])
            id.append(item[2])
            label.append(item[1])

        label = torch.LongTensor(label)

        data_batch_tensor = torch.stack(data, dim=0)

        return data_batch_tensor, label, id


def get_dsprites_train_loader(dir='/home/sachahai/Documents/VAE_manifold/Code/',
                              val_split=0.9, shuffle=True, seed=42, batch_size=64):
    # img_size = 64
    path = os.path.join(dir, 'human_guidance')
    dataset = DisentangledSpritesDataset(path, transform=transforms.Compose([
        transforms.ToTensor(),  # Will rescale to 0-1 Float32
        Double_to_Float()
    ])
                                         )

    train_idx, valid_idx = train_test_split(np.arange(len(dataset)),
                                            test_size=val_split,
                                            shuffle=shuffle, random_state=seed)

    # Create data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=12)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=12)

    return train_loader, val_loader


def get_dsprites_inference_loader(dir='/home/sachahai/Documents/VAE_manifold/Code/', shuffle=False, batch_size=64):
    # img_size = 64
    path = os.path.join(dir, 'human_guidance')
    dataset = DisentangledSpritesDataset(path, transform=transforms.Compose([
        transforms.ToTensor(),  # Will rescale to 0-1 Float32
        Double_to_Float()
    ]), train=False
                                         )

    dataloaders = DataLoader(dataset, batch_size=batch_size, collate_fn=Desprite_ID_Collator(), shuffle=shuffle, num_workers=12)

    return dataset, dataloaders


def get_MNIST_dataloader(path='/mnt/Linux_Storage/', batch_size=128):
    train_dataset = torchvision.datasets.MNIST(path, train=True, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.Resize(32),
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       (0.1307,), (0.3081,))
                                               ]))

    train_dataset = torch.utils.data.random_split(train_dataset, [20000, len(train_dataset) - 20000])[0]

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


if __name__ == '__main__':
    _, loader = get_dsprites_inference_loader(val_split=0.05)
    trainiter = iter(loader)
    features, _ = next(trainiter)
    _, _ = imshow_tensor(features[0])
