# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-03T11:51:36+11:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-21T11:28:19+10:00

'''
This file contains classes and function that are usefull to load the raw data,
organize it throughout batch, and preprocess it to be fed to a VAE network

DatasetFolder are used. Dataset should be arranged in subfolders that represent
the (expected) class/cluster idendity. VAEs are unsupervised methods, the latter
will only be used as visual colored labels once the model is trained.
'''
###############################
####### Imports ###############
###############################
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

from skimage import io
from skimage.util import img_as_float, pad
from PIL import Image
from skimage.transform import resize, rotate
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import random
from copy import copy


###############################
#### DataLoader ###############
###############################

#Training dataloader
def get_train_val_dataloader(root_dir,input_size,batchsize,test_split=0.2):
    """
    From a unique folder that contains the whole dataset, divided in different subfolders
    related to class identity, return a train and validation dataloader that can be used
    to train a VAE network.
    Note that a stratified splitting is performed, ensuring that class proportion is maintained
    is both sets.

    Params :
        - root_dir : path to the folder containing the dataset
        - input_size : imgs will all be input_size x input_size (rescale or pad)
        - test_split : Proportion of sample (0-1) that will be part of validation set
    """
    trsfm = image_tranforms(input_size)
    dataset = datasets.DatasetFolder(root=root_dir,loader=load_from_path(),extensions=('.png','.jpg','.tif','.tiff'), transform=trsfm)
    targets = dataset.targets

    #Stratify splitting
    train_idx, valid_idx=train_test_split(np.arange(len(targets)),
                                            test_size=test_split,
                                            shuffle=True,
                                            stratify=targets)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    dataset_train = dataset
    dataset_valid = copy(dataset) #To enable different transforms
    dataset_valid.transform = transforms.Compose([
        zPad_or_Rescale(input_size),
        transforms.ToTensor(),
        Double_to_Float()])
    dataset_train.transform = trsfm
    # Use the following lines to NOT use data augmentation
    # dataset_train.transform = transforms.Compose([
    #     zPad_or_Rescale(input_size),
    #     transforms.ToTensor(),
    #     Double_to_Float()])
    #Create DataIterator, yield batch of img and label easily and in time, to not load full heavy set
    #Dataloader iterators
    train_loader = DataLoader(dataset_train, batch_size=batchsize,sampler=train_sampler,drop_last=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batchsize,sampler=valid_sampler,drop_last=True)

    return train_loader, valid_loader

#Inference dataloader (no data augmentation and no train/test split)
def get_inference_dataset(dataset_dir, batchsize, input_size, shuffle=False, droplast=False):
    '''
    DataLoader for inference. No data augmentation and no train/test split
    From a unique folder that contains the whole dataset, divided in different subfolders
    related to class identity

    Params :
        - root_dir : path to the folder containing the dataset
        - input_size : imgs will all be input_size x input_size (rescale or pad)
    Returns :
        - (DatasetFolder) :
        - (DatasetLoader) :
    '''
    inference_trfm = transforms.Compose([
        #Data arrive as HxWxC float64 0 - 1.0 ndarray
        zPad_or_Rescale_inference(input_size), # Rescale and Pad to fixed size
        ToTensor_inference(), #Will rescale to 0-1 Float32
        Double_to_Float_inference()])

    data = datasets.DatasetFolder(root=dataset_dir,loader=keep_Metadata_from_path(),extensions=('.png','.jpg','.tif','.tiff'), transform=inference_trfm)
    dataloaders = DataLoader(data, batch_size=batchsize, collate_fn=My_ID_Collator(), shuffle=shuffle,drop_last=droplast)

    return data, dataloaders



###############################
#### Custom DataLoader Utils ##
###############################

#Need to specify custom loader for a DataSetFolder, Because ImageFolder CAN'T be used,
#because it opens image as PIL object, which do not support uint16...
class load_from_path(object):
    """Load an image from its path
    Samples are returned as ndarray, float64 0-1"""
    def __call__(self, path):

        sample = io.imread(path,plugin='tifffile') #load H x W x C tiff files
        #Single cell image are in uint8, 0-255
        sample = img_as_float(sample) # float64 0 - 1.0
        return sample

class keep_Metadata_from_path(object):
    """Load an image from its path, but keep track of the cell id at the same time.
    It enables therefore to inspect some characteristic of the built
    latent reprensetation alongside some metadata information of the dataset.
    Samples are returned as ndarray, float64 0-1"""

    def __call__(self, path):

        sample = io.imread(path,plugin='tifffile') #load H x W x C tiff files
        sample = img_as_float(sample) # float64 0 - 1.0
        #Extract the unique cell id from the file name
        file_name = path.split('/')

        return (sample, file_name[-1])

#Useful for inference dataset that need to keep track of cell ID
class My_ID_Collator(object):
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
        data = [item[0][0] for item in batch]
        file_name = [item[0][1] for item in batch]
        target = [item[1] for item in batch]
        target = torch.LongTensor(target)

        data_batch_tensor = torch.stack(data,dim=0)

        return data_batch_tensor, target, file_name


##############################################
#### Imgs Transformation / Augmentation ##
##############################################

def image_tranforms(input_size):
    """
    Basic preprocessing that will be apply on data throughout dataloader.
    Image are resize (zero/padded or reshape) to a given input size,
    data augmentation is performed and image are translated to tensor

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
    img_transforms = transforms.Compose([
            zPad_or_Rescale(input_size),
            # 2) Data augmentation
            RandomRot90(),
            RandomSmallRotation(),
            RandomVFlip(),
            RandomHFlip(),
            transforms.ToTensor(), #Will rescale to 0-1 Float32
            Double_to_Float()
        ])
    return img_transforms


#Create callable class to have custom tranform of our data
class zPad_or_Rescale(object):
    """Resize all data to fixed size of input_size
    if any dimension is bigger than input_size -> RESCALE
    if both dimension are smaller than input_size -> Zero Pad

    Image is returned as an ndarray float64 0-1"""
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, sample):
        img_arr = sample #HxWxC

        h = img_arr.shape[0]
        w = img_arr.shape[1]

        fixed_size = (self.input_size,self.input_size)

        if ((h > fixed_size[0]) or (w > fixed_size[1])):
            # Resize
            img_resized = resize(img_arr,fixed_size,preserve_range=False, anti_aliasing=True)
            #anti-aliasing import before down sampling
        elif ((h <= fixed_size[0]) and (w <= fixed_size[1])):
            # ZeroPad
            diff_h = fixed_size[0] - h
            diff_w = fixed_size[1] - w
            img_resized = pad(img_arr,((int(np.round(diff_h/2.)),diff_h-int(np.round(diff_h/2.))),(int(np.round(diff_w/2.)),diff_w-int(np.round(diff_w/2.))),(0,0)))
        assert img_resized[:,:,0].shape == fixed_size, "Error in padding / rescaling"

        return img_resized

class zPad_or_Rescale_inference(object):
    """Resize all data to fixed size of 256x256
    if any dimension is bigger than 256 -> RESCALE
    if both dimension are smaller than 256 -> Zero Pad

    Image is returned as an ndarray float64 0-1"""
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, sample):

        image, file_name = sample[0], sample[1]
        img_arr = image #HxWxC

        h = img_arr.shape[0]
        w = img_arr.shape[1]

        fixed_size = (self.input_size,self.input_size)

        if ((h > fixed_size[0]) or (w > fixed_size[1])):
            # Resize
            img_resized = resize(img_arr,fixed_size,preserve_range=False, anti_aliasing=True)
            #anti-aliasing import before down sampling
        elif ((h <= fixed_size[0]) and (w <= fixed_size[1])):
            # ZeroPad
            diff_h = fixed_size[0] - h
            diff_w = fixed_size[1] - w

            img_resized = pad(img_arr,((int(np.round(diff_h/2.)),diff_h-int(np.round(diff_h/2.))),(int(np.round(diff_w/2.)),diff_w-int(np.round(diff_w/2.))),(0,0)))

        assert img_resized[:,:,0].shape == fixed_size, "Error in padding / rescaling"

        return (img_resized, file_name)


class RandomVFlip(object):
    """Data augmentation
    Randomly flip the image horizontally"""
    def __call__(self, sample):
        if random.random() < 0.5:
            sample = np.flipud(sample)

        return np.ascontiguousarray(sample)

class RandomHFlip(object):
    """Data augmentation
    Randomly flip the image vertically"""
    def __call__(self, sample):
        if random.random() < 0.5:
            sample = np.fliplr(sample)

        return np.ascontiguousarray(sample)

class RandomSmallRotation(object):
    """Data augmentation
    Randomly rotate the image"""
    def __call__(self, sample):
        rot_angle = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85]

        if random.random() < 0.5:
            sample = rotate(sample,np.random.choice(rot_angle))

        return sample

class RandomRot90(object):
    """Data augmentation
    Randomly rotate the image by 90 degree"""
    def __call__(self, sample):
        if random.random() < 0.5:
            sample = np.rot90(sample,k=1,axes=(0,1))

        return np.ascontiguousarray(sample)

class Double_to_Float(object):
    '''Final inputs need to be Float'''
    def __call__(self, sample):

        return sample.float()

class Double_to_Float_inference(object):
    '''Final inputs need to be Float'''
    def __call__(self, sample):

        image, file_name = sample[0], sample[1]
        return (image.float(), file_name)

class ToTensor_inference(object):
    '''From ndarray to Tensor'''
    def __call__(self,sample):
        image, file_name = sample[0], sample[1]

        image = image.transpose((2,0,1))

        return (torch.from_numpy(image), file_name)


def imshow_tensor(tensor_img, ax = None):
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
