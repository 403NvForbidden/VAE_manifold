# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-05-25T09:37:50+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-07-03T16:01:28+10:00

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
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

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
            loss = criterion(outputs, labels.long())
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
        self.low_dim_names = low_dim_names

    def __len__(self):
        return len(self.latentCode_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        x_coord = self.latentCode_frame.loc[idx,self.low_dim_names[0]]
        y_coord = self.latentCode_frame.loc[idx,self.low_dim_names[1]]
        z_coord = self.latentCode_frame.loc[idx,self.low_dim_names[2]]

        #samples = np.array(list(zip(x_coord,y_coord,z_coord))) # batch x 3
        sample = np.array([x_coord,y_coord,z_coord])
        label = self.latentCode_frame.loc[idx,self.GT_class]

        if self.transform:
            sample = self.transform(sample)

        return sample, label



def classifier_performance(path_to_csv,low_dim_names=['x_coord','y_coord','z_coord'],Metrics=[True,False,False],num_iteration=5):
    '''
    Given a CSV-file containing a 3D latent code to evaluate, built a simple
    (200 unit single hidden layer) NN classifier. Test accuracy can be used as
    a numerical value to assess the quality of the latent code.
    High accuracy -> the latent dimensions present the data in a way where ground
    truth cluster can easily be discriminate

    Return the classifiier performance over num_iteration (to compute mean and std)

    Metric 1 : Test classification performance on all single cell except uniform cluster (7)
    Metric 2 : Same, but only strong phenotype change (>0.5)
    Metric 3 : Same, but meta-cluster. Discriminate between 1&2, 3&4 and 5&6
    '''

    if isinstance(path_to_csv,str):
        latentCode_frame = pd.read_csv(path_to_csv)
    else:
        latentCode_frame = path_to_csv

    latentCode_frame['GT_label'] = pd.to_numeric(latentCode_frame['GT_label'])
    latentCode_frame['GT_label'] = latentCode_frame['GT_label'].astype(int)

    #################################
    ####### Metric One ##############
    #################################
    ## All cells included, except cluster 7
    if Metrics[0]:
        non_last_cluster = latentCode_frame['GT_label'] != 7
        latentCode_frame = latentCode_frame[non_last_cluster]
        #For NN, the class must be between 0 - num_class-1
        latentCode_frame['GT_label'] = latentCode_frame['GT_label'].subtract(1)

        ##Make a half / half train test split that have the same percentage of classes as original dataset
        labels = latentCode_frame['GT_label'].values
        train_test_split = StratifiedShuffleSplit(n_splits=num_iteration,test_size=0.5,random_state=12) #Change n_splits if want to have several run

        train_acc = []
        test_acc = []
        perclass_te_acc = []
        #For statistical relevance, make several train-test split
        for train_index, test_index in train_test_split.split(np.zeros((len(latentCode_frame),3)),labels):
            dataset_train = latentCode_frame.iloc[train_index]
            dataset_test = latentCode_frame.iloc[test_index]
            dataset_train.reset_index(inplace=True)
            dataset_test.reset_index(inplace=True)

            #Built train and test dataset/dataloader
            tr_dataset = Dataset_from_csv(dataset_train,'GT_label',low_dim_names=low_dim_names)
            tr_dataloader = DataLoader(tr_dataset,batch_size=128,shuffle=True)
            te_dataset = Dataset_from_csv(dataset_test,'GT_label',low_dim_names=low_dim_names)
            te_dataloader = DataLoader(te_dataset,batch_size=128,shuffle=True)

            model_1 = Classifier_Net()
            model_1 = model_1.float()
            model_1 = model_1.cuda()

            #train on train_dataloader
            train_net(model_1,20,tr_dataloader)

            #train and test accuracy
            train_acc.append(perf_eval(model_1,tr_dataloader))
            test_acc.append(perf_eval(model_1,te_dataloader))

            perclass_te_acc.append(perf_eval_perclass(model_1,te_dataloader))

        return train_acc, test_acc, perclass_te_acc

    #################################
    ####### Metric two ##############
    #################################
    ## Disregard cluster 7, and only consider strong phenotypic change (>0.5)
    if Metrics[1]:
        non_last_cluster = latentCode_frame['GT_label'] != 7
        strong_phenotype = latentCode_frame['GT_dist_toInit_state']>=0.5
        latentCode_frame = latentCode_frame[non_last_cluster & strong_phenotype]
        #For NN, the class must be between 0 - num_class-1
        latentCode_frame['GT_label'] = latentCode_frame['GT_label'].subtract(1)

        ##Make a half / half train test split that have the same percentage of classes as original dataset
        labels = latentCode_frame['GT_label'].values
        train_test_split = StratifiedShuffleSplit(n_splits=num_iteration,test_size=0.5,random_state=12) #Change n_splits if want to have several run

        train_acc = []
        test_acc = []
        perclass_te_acc = []
        #For statistical relevance, make several train-test split
        for train_index, test_index in train_test_split.split(np.zeros((len(latentCode_frame),3)),labels):
            dataset_train = latentCode_frame.iloc[train_index]
            dataset_test = latentCode_frame.iloc[test_index]
            dataset_train.reset_index(inplace=True)
            dataset_test.reset_index(inplace=True)

            #Built train and test dataset/dataloader
            tr_dataset = Dataset_from_csv(dataset_train,'GT_label',low_dim_names=low_dim_names)
            tr_dataloader = DataLoader(tr_dataset,batch_size=128,shuffle=True)
            te_dataset = Dataset_from_csv(dataset_test,'GT_label',low_dim_names=low_dim_names)
            te_dataloader = DataLoader(te_dataset,batch_size=128,shuffle=True)

            model_2 = Classifier_Net()
            model_2 = model_2.float()
            model_2 = model_2.cuda()

            #train on train_dataloader
            train_net(model_2,20,tr_dataloader)

            #train and test accuracy
            train_acc.append(perf_eval(model_2,tr_dataloader))
            test_acc.append(perf_eval(model_2,te_dataloader))

            perclass_te_acc.append(perf_eval_perclass(model_2,te_dataloader))

        return train_acc, test_acc, perclass_te_acc

    #################################
    ####### Metric three ##############
    #################################
    ## Disregard cluster 7, only strong phenotypic change (>0.5), and METACLUSTER
    # 1&2 vs 3&4 vs 5&6
    if Metrics[2]:
        non_last_cluster = latentCode_frame['GT_label'] != 7
        strong_phenotype = latentCode_frame['GT_dist_toInit_state']>=0.5
        latentCode_frame = latentCode_frame[non_last_cluster & strong_phenotype]
        #For NN, the class must be between 0 - num_class-1
        latentCode_frame['GT_label'] = latentCode_frame['GT_label'].subtract(1)

        #Built metacluster (1&2 vs 3&4 vs 5&6)
        latentCode_frame['GT_label'].replace(1,0,inplace=True)
        latentCode_frame['GT_label'].replace(2,1,inplace=True)
        latentCode_frame['GT_label'].replace(3,1,inplace=True)
        latentCode_frame['GT_label'].replace(4,2,inplace=True)
        latentCode_frame['GT_label'].replace(5,2,inplace=True)

        ##Make a half / half train test split that have the same percentage of classes as original dataset
        labels = latentCode_frame['GT_label'].values
        train_test_split = StratifiedShuffleSplit(n_splits=num_iteration,test_size=0.5,random_state=12) #Change n_splits if want to have several run

        train_acc = []
        test_acc = []
        perclass_te_acc = []
        #For statistical relevance, make several train-test split
        for train_index, test_index in train_test_split.split(np.zeros((len(latentCode_frame),3)),labels):
            dataset_train = latentCode_frame.iloc[train_index]
            dataset_test = latentCode_frame.iloc[test_index]
            dataset_train.reset_index(inplace=True)
            dataset_test.reset_index(inplace=True)

            #Built train and test dataset/dataloader
            tr_dataset = Dataset_from_csv(dataset_train,'GT_label',low_dim_names=low_dim_names)
            tr_dataloader = DataLoader(tr_dataset,batch_size=128,shuffle=True)
            te_dataset = Dataset_from_csv(dataset_test,'GT_label',low_dim_names=low_dim_names)
            te_dataloader = DataLoader(te_dataset,batch_size=128,shuffle=True)

            model_3 = Classifier_Net(num_of_class=3)
            model_3 = model_3.float()
            model_3 = model_3.cuda()

            #train on train_dataloader
            train_net(model_3,20,tr_dataloader)

            #train and test accuracy
            train_acc.append(perf_eval(model_3,tr_dataloader))
            test_acc.append(perf_eval(model_3,te_dataloader))

            perclass_te_acc.append(perf_eval_perclass(model_3,te_dataloader))

        return train_acc, test_acc, perclass_te_acc


def compare_models(list_of_csv,Metrics=[True,False,False],num_iteration=5):

    model_names=[]
    all_means=[]
    all_stds=[]

    for i, csv_file in enumerate(list_of_csv):

        train_acc, test_acc, perclass_te_acc = classifier_performance(csv_file,Metrics,num_iteration)

        #In future, probably don-t care about train accuracy
        all_means.append(np.mean(train_acc))
        all_stds.append(np.std(train_acc))
        model_names.append(f'Model {i+1} -Train')
        all_means.append(np.mean(test_acc))
        all_stds.append(np.std(test_acc))
        model_names.append(f'Model {i+1} -Test')


        perclass_means = np.mean(np.array(perclass_te_acc),axis=0)
        perclass_stds = np.std(np.array(perclass_te_acc),axis=0)

        #One plot per model for the per-class accuracy
        names = [f'cluster {j}' for j in range(1,len(perclass_means)+1)]
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(names)),perclass_means,yerr=perclass_stds,align='center',alpha=0.5, ecolor='black',capsize=10)
        ax.set_ylabel('Test classification accuracy [%]')
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names)
        ax.set_title(f'Per Class accuracy of model {i+1}, avg over 10 runs')
        ax.yaxis.grid(True)

        plt.tight_layout()
        #plt.savefig()
        plt.show()

    fig2, ax2 = plt.subplots(figsize=(10,6))
    ax2.bar(np.arange(len(model_names)),all_means,yerr=all_stds,align='center',alpha=0.5, ecolor='black',capsize=10)
    ax2.set_ylabel('Classification accuracy [%]')
    ax2.set_xticks(np.arange(len(model_names)))
    ax2.set_xticklabels(model_names)
    ax2.set_title('NN Classifier acc (avg over 10 runs) as latent code performance metric')
    ax2.yaxis.grid(True)

    plt.tight_layout()
    #plt.savefig()
    plt.show()
