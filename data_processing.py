# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-03T11:51:36+11:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahaidinger
# @Last modified time: 2020-04-06T13:01:15+10:00

'''
This file contains classes and function that are usefull to load the raw data,
organize it throughout batch, and preprocess it to be fed to a VAE network
'''

from torchvision import datasets
from torch.utils.data import DataLoader
from skimage import io
from skimage.util import img_as_ubyte, pad
from PIL import Image
from torchvision import transforms
from skimage.transform import resize

import matplotlib.pyplot as plt
import numpy as np

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
        'train': DataLoader(data['train'], batch_size=batchsize, shuffle=True),
        'val': DataLoader(data['val'], batch_size=batchsize, shuffle=True),
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
    """Load an image from its path"""

    def __call__(self, path):

        sample = io.imread(path,plugin='tifffile') #load H x W x 4 tiff files
        #Single cell image are already in uint8 (PIL supported format)
        #sample = img_as_ubyte(sample) # uint8 0-255

        #plt.imshow(sample)
        #plt.title(f'Raw 1-channel (actin) single image, shape : {sample.shape}')
        #plt.show()

        #Mimic the 4 channels data that we will have in the end (# TODO: REMOVE)
        img_4chan = np.repeat(sample[:,:,np.newaxis], repeats=4, axis=2)
        #convert it to PIL image to allow PyTorch transforms manipulation
        sample = Image.fromarray(img_4chan) #uint 0-255


        return sample

def image_tranforms():
    """
    Basic preprocessing that will be apply on data throughout dataloader.
    Apply this pipeline if dataset is already in PATCH (64,64), or other.
    Few/None random transformation because patch already extracted
    Mainly : Patch are bring to RGB, translate to tensor and then normalize according
    to ImageNet Standard (Range of value that each channel expect to see)

    Parameters
    ----------
    None

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
            #Data arrive as HxWx4 uint8 (0-255) image, open with PIL
            # 1) Rescale and Pad to fixed
            zPad_or_Rescale(),
            # 2) Data augmentation with Pil image
            transforms.RandomHorizontalFlip(),
            # ROTATION ? WARP ? NOISE ?
            # 2) Data augmentation with Pil image
            transforms.ToTensor(), #Will rescale to 0-1 Float32

        ]),
        # Validation does not use augmentation
        'val':
        transforms.Compose([
            zPad_or_Rescale(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        # Test does not use augmentation
        'test':
        transforms.Compose([
            zPad_or_Rescale(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
    }

    return img_transforms


#Create callable class to have custom tranform of our data
class zPad_or_Rescale(object):
    """Resize all data to fixed size of 128x128
    if any dimension is bigger than 128 -> RESCALE
    if both dimension are smaller than 128 -> Zero Pad"""

    def __call__(self, sample):
        img = sample #PIL img
        img_arr = np.array(img) #HxWx4

        h = img_arr.shape[0]
        w = img_arr.shape[1]

        fixed_size = (128,128)

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

        #go back to unint8 to be able to open as a PIL image
        img_resized = img_as_ubyte(img_resized) # uint8 0-255
        # TODO:  REALLY consider if using PIL is necessary... data augm can be performed easily by hand !
        #Retur to PIL img
        img_pil = Image.fromarray(img_resized)

        return img_pil


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

    ax.imshow(image[:,:,0])
    plt.title('Single_cell image resized to (128x128)')
    #plt.axis('off')

    return ax, image
