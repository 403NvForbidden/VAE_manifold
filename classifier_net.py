# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-05-25T09:37:50+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-07-01T17:28:42+10:00

'''
File containing the small FC network that is used as a
performance metric of learnt latent code
'''

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import cuda
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Classifier_Net(nn.Module):
    def __init__(self, zdim=3, num_of_class=6):
        super(Classifier_Net, self).__init__()

        self.zdim = zdim
        self.num_of_class = num_of_class

        self.fc1 = nn.Linear(zdim, 200)
        self.fc2 = nn.Linear(200, num_of_class)

    def forward(self, x):
        x = (F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_net(model,epochs,trainloader):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    history_loss = []

    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].float().cuda(), data[1].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            history_loss.append(loss.item())
            if i % 10 == 9:    # print every 2000 mini-batches
                #print('[%d, %5d] loss: %.3f' %
                #      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    #print('Finished Training')

def perf_eval(model,testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].float().cuda(), data[1].cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))
    return 100 * correct / total

def perf_eval_perclass(model,testloader):
    class_correct = list(0. for i in range(model.num_of_class))
    class_total = list(0. for i in range(model.num_of_class))
    per_class_acc = []
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].float().cuda(), data[1].cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(c)):
                label = labels[i]
                class_correct[label] += c[i].item() #add 1 only if predict true class
                class_total[label] += 1
    # for i in range(model.num_of_class):
    #     print('Accuracy of class %5s : %2d %%' % (
    #         i, 100 * class_correct[i] / class_total[i]))
    #     per_class_acc.append(100 * class_correct[i] / class_total[i])
    return per_class_acc




class Dataset_from_csv(Dataset):
    """Latent Code dataset."""

    def __init__(self, latentCode_frame, GT_class,low_dim_names=['x_coord','y_coord','z_coord'], transform=None):
        """
        Args:
            latentCode_frame (pandas DataFrame): DataFrame containing latent code and class
            GT_class (string) : Name of the column in which ground truth class are stored
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.latentCode_frame = latentCode_frame
        self.transform = transform
        self.GT_class = GT_class

    def __len__(self):
        return len(self.latentCode_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        x_coord = self.latentCode_frame.loc[idx,low_dim_names[0]]
        y_coord = self.latentCode_frame.loc[idx,low_dim_names[1]]
        z_coord = self.latentCode_frame.loc[idx,low_dim_names[2]]

        #samples = np.array(list(zip(x_coord,y_coord,z_coord))) # batch x 3
        sample = np.array([x_coord,y_coord,z_coord])
        label = self.latentCode_frame.loc[idx,self.GT_class]

        if self.transform:
            sample = self.transform(sample)

        return sample, label
