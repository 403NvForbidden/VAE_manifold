# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-03T11:51:36+11:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-04-20T09:09:57+10:00

'''
This file contains classes and function that are usefull to load the raw data,
organize it throughout batch, and preprocess it to be fed to a VAE network
'''

from torchvision import datasets
from torch.utils.data import DataLoader
from skimage import io
from skimage.util import img_as_float, pad
from PIL import Image
from torchvision import transforms
from skimage.transform import resize

import matplotlib.pyplot as plt
import numpy as np
import random

def get_dataloader(root_dir,img_transforms,batchsize):
    """
    Return an appropriate DataLoader, to load data in batch and apply the desired
    pipeline preprocess
    """
    data = {
        'train':
        datasets.DatasetFolder(root=root_dir[0],loader=load_from_path(),extensions=('.png','.jpg','.tif','.tiff'), transform=img_transforms['train']),
        'val':
        datasets.DatasetFolder(root=root_dir[1],loader=load_from_path(),extensions=('.png','.jpg','.tif','.tiff'), transform=img_transforms['val']),
        #'test':
        #datasets.ImageFolder(root=root_dir[2], transform=img_transforms['test'])
    }

    #Create DataIterator, yield batch of img and label easily and in time, to not load full heavy set
    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=batchsize, shuffle=True, drop_last=True),
        'val': DataLoader(data['val'], batch_size=batchsize, shuffle=True, drop_last=True),
        #'test': DataLoader(data['test'], batch_size=batchsize, shuffle=True)
    }

    return data, dataloaders
#How to use it :
#trainiter = iter(dataloaders['train'])
#features, labels = next(trainiter)
# The shape of a batch is (batch_size, color_channels, height, width)


#Need to specify custom loader for a DataSetFolder, Because ImageFolder CAN'T be used,
#because it opens image as PIL object, which do not support uint16...
class load_from_path(object):
    """Load an image from its path
    Samples are returned as ndarray, float64 0-1"""

    def __call__(self, path):

        sample = io.imread(path,plugin='tifffile') #load H x W x 4 tiff files
        #Single cell image are in uint8, 0-255

        sample = img_as_float(sample) # float64 0 - 1.0

        return sample

def image_tranforms(input_size):
    """
    Basic preprocessing that will be apply on data throughout dataloader.
    Apply this pipeline if dataset is already in PATCH (64,64), or other.
    Few/None random transformation because patch already extracted
    Mainly : Patch are bring to RGB, translate to tensor and then normalize according
    to ImageNet Standard (Range of value that each channel expect to see)

    Parameters
    ----------
    input_size (int) : Image will be reshaped C x input_size x input_size

    Returns
    -------
    patch_tranforms : torch.transforms.Compose
        Object that link several transformations in a pipeline process

    """

    # %% Bulding a tranformation pipeline on Pytorch thanks to Compose
    # Image transformations
    img_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            #Data arrive as HxWx4 float64 0 - 1.0 ndarray
            # 1) Rescale and Pad to fixed
            zPad_or_Rescale(input_size),
            # 2) Data augmentation
            RandomHandVFlip(),
            # ROTATION ? WARP ? NOISE ?
            transforms.ToTensor(), #Will rescale to 0-1 Float32
            Double_to_Float()

        ]),
        # Validation does not use augmentation
        'val':
        transforms.Compose([
            zPad_or_Rescale(input_size),
            transforms.ToTensor(),
            Double_to_Float()
        ]),
        # Test does not use augmentation
        'test':
        transforms.Compose([
            zPad_or_Rescale(input_size),
            transforms.ToTensor(),
            Double_to_Float()
        ]),
    }

    return img_transforms


#Create callable class to have custom tranform of our data
class zPad_or_Rescale(object):
    """Resize all data to fixed size of 256x256
    if any dimension is bigger than 256 -> RESCALE
    if both dimension are smaller than 256 -> Zero Pad

    Image is returned as an ndarray float64 0-1"""
    def __init__(self, input_size):
        self.input_size = input_size


    def __call__(self, sample):
        img_arr = sample #HxWx4

        h = img_arr.shape[0]
        w = img_arr.shape[1]

        fixed_size = (self.input_size,self.input_size)

        if ((h > fixed_size[0]) or (w > fixed_size[1])):
            # Resize
            img_resized = resize(img_arr,fixed_size,preserve_range=False, anti_aliasing=True)
            # TODO: CHeck if ok to stay in uint8 here
            #anti-aliasing import before down sampling
        elif ((h <= fixed_size[0]) and (w <= fixed_size[1])):
            # ZeroPad
            diff_h = fixed_size[0] - h
            diff_w = fixed_size[1] - w

            img_resized = pad(img_arr,((int(np.round(diff_h/2.)),diff_h-int(np.round(diff_h/2.))),(int(np.round(diff_w/2.)),diff_w-int(np.round(diff_w/2.))),(0,0)))

        assert img_resized[:,:,0].shape == fixed_size, "Error in padding / rescaling"

        return img_resized


class RandomHandVFlip(object):
    """Data augmentation
    Randomly flip the image verticallz or horizontally"""

    def __call__(self, sample):

        if random.random() < 0.5:
            sample = np.fliplr(sample)

        if random.random() < 0.5:
            sample = np.flipud(sample)

        ### TODO:  Do the same with np.rot() ???

        return np.ascontiguousarray(sample)

class Double_to_Float(object):

    def __call__(self, sample):

        return sample.float()


def imshow_tensor(tensor_img, ax = None, tittle = None):
    """
    Display a tensor as an image. Swap of channels.

    Parameters
    ----------
    tensor_img : torch.Tensor
        PyTorch Tensors that is wanted to be displayed

    Returns
    -------
    ax : matplotlib.pyplot.Axes
        axes on which img is plotted
    image : ndarray
        returned image, to be displayed

    """
    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = tensor_img.numpy().transpose((1, 2, 0))

    ax.imshow(image[:,:,0:3])
    inSize = image.shape[0]
    plt.title(f'Single_cell image resized to ({inSize}x{inSize})')
    #plt.axis('off')

    return ax, image
