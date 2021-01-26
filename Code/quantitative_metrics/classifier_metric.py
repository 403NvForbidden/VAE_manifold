# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-05-25T09:37:50+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-30T12:22:23+10:00

'''
Supervised classifier accuracy as a metric to gauge the quality of learnt representation
It will gauge the ability of the projection methods to extract features that enable easy
discrimination of the ground truth classes.

The file contains all the functions and classess to create and trained a small NN
to predict class identity from the latent codes.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import StratifiedShuffleSplit

import warnings

warnings.filterwarnings('ignore')


class Classifier_Net(nn.Module):
    '''
    Small NN of fixed capacity that predict class identity from the latent codes
    '''

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


def train_net(model, epochs, trainloader, num_class=6, imbalance_weight_horvath=False):
    '''
    Train a simple NN classifer to predict class identity from latent code

    Params :
        model (nn.Module) : Simple FC NN to train
        epochs (int) : Number of epochs of the training procedure
        trainloader (DataLoader) : Dataloader use for training
        num_class (int) : Number of different class in the dataset
        imbalance_weight_horvath (bolean) : Set to True if Horvath or Chaffer Dataset
                The loss will be weighted to take into account the imbalanced proportion of classes
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

    if imbalance_weight_horvath:  # Horvath dataset2 is imbalanced. Weight the loss accordingly
        if num_class == 6:  # Horvath
            weights = [0.085, 0.085, 0.085, 0.25, 0.61, 1.]
        elif num_class == 12:  # Chaffer
            weights = [0.27, 0.031, 0.5, 1., 0.031, 0.41, 0.51, 0.51, 0.27, 0.27, 0.64, 0.28]
        class_weights = torch.FloatTensor(weights).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weights)

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

            running_loss += loss.item()
            history_loss.append(loss.item())
            if i % 10 == 9:  # print every 2000 mini-batches
                # print('[%d, %5d] loss: %.3f' %
                #      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        if epochs % 5 == 0:
            print("============>", epochs)


def perf_eval(model, testloader):
    '''
    Evaluate the classifier accuracy of a trained classifier Network.

    Params :
        model (nn.Module) : A trained NN classifier
        testloader (DataLoader) : Dataloader that load test samples

    Return a single scalar, the classifier test accuracy
    '''
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


def perf_eval_perclass(model, testloader):
    '''
    Evaluate the per_class classifier accuracy of a trained classifier Network.

    Params :
        model (nn.Module) : A trained NN classifier
        testloader (DataLoader) : Dataloader that load test samples

    Return a list of all per_class accuracy
    '''
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
                class_correct[label] += c[i].item()  # add 1 only if predict true class
                class_total[label] += 1
    # for i in range(model.num_of_class):
    #     print('Accuracy of class %5s : %2d %%' % (
    #         i, 100 * class_correct[i] / class_total[i]))
    #     per_class_acc.append(100 * class_correct[i] / class_total[i])
    return per_class_acc


class Dataset_from_csv(Dataset):
    """
    Custom Dataset, that enable to load latent codes and ground truth information (Class ID)
    from a CSV file in which they are saved.
    Class identity need to be stored under a column named 'GT_label' and should be an int between 1 - num_class
    """

    def __init__(self, latentCode_frame, GT_class, low_dim_names=['x_coord', 'y_coord', 'z_coord'], transform=None):
        """
        Args:
            latentCode_frame (pandas DataFrame): DataFrame containing latent code and class
            GT_class (string) : Name of the column in which ground truth class are stored
            low_dim_names ([string]) : Names of the columns that stores the latent codes
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

        x_coord = self.latentCode_frame.loc[idx, self.low_dim_names].values
        # y_coord = self.latentCode_frame.loc[idx, self.low_dim_names[1]]
        # z_coord = self.latentCode_frame.loc[idx, self.low_dim_names[2]]

        sample = np.array(x_coord, dtype=float)
        label = self.latentCode_frame.loc[idx, self.GT_class]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


def classifier_performance(path_to_csv, low_dim_names=['x_coord', 'y_coord', 'z_coord'], Metrics=[True, False, False],
                           num_iteration=5, num_class=6, class_to_ignore=7, imbalanced_data=False):
    '''
    Given a CSV-file containing a 3D latent code to evaluate, built a simple
    (200 unit single hidden layer) NN classifier. Test accuracy can be used as
    a numerical value to assess the quality of the latent code.
    High accuracy -> the latent dimensions present the data in a way where ground
    truth cluster can easily be discriminate

    Return the classifiier performance over num_iteration (to compute mean and std)

    Metric 1 : Test classification performance on all single cell
    Metric 2 : Same, but only strong phenotype change (>0.5)
    Metric 3 : Same, but meta-cluster. Discriminate between 1&2, 3&4 and 5&6
    Guidelines : Use Metric 1 for all dataset except BBBC dataset that can use all 3 metrics

    Params :
        path_to_csv (string or DataFrame) : Path to csv file or directly the DataFrame that contains the latent codes and ground truth
        low_dim_names ([string]) : Names of the columns that stores the latent codes
        Metrics ([bolean, bolean, bloean]) : Which metric to compute. [True, False, False] will compute metric 1
        num_iteration (int) : The whole process (new model, new train-test split, new training) will be performed num_iteration time to obtain a mean and std
        num_class (int) : Number of different class in the dataset
        class_to_ignore ('None' or int) : If 'None', no class are ignored. If a int, this class ID will be ignored. Guideline : Ignore class 7 for BBBC dataset
        imbalanced_data (bolean) : Set to True if Horvath or Chaffer Dataset
                        The loss will be weighted to take into account the imbalanced proportion of classes

    Return train, test and per_class test accuracies
    '''

    if isinstance(path_to_csv, str):
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
        if class_to_ignore != 'None':
            non_last_cluster = latentCode_frame['GT_label'] != class_to_ignore
            latentCode_frame = latentCode_frame[non_last_cluster]
        # For NN, the class must be between 0 - num_class-1
        latentCode_frame['GT_label'] = latentCode_frame['GT_label'].subtract(1)

        ##Make a half / half train test split that have the same percentage of classes as original dataset
        labels = latentCode_frame['GT_label'].values
        train_test_split = StratifiedShuffleSplit(n_splits=num_iteration, test_size=0.5,
                                                  random_state=12)  # Change n_splits if want to have several run

        model_1 = Classifier_Net(zdim=len(low_dim_names), num_of_class=num_class).cuda()

        train_acc = []
        test_acc = []
        perclass_te_acc = []
        # For statistical relevance, make several train-test split
        for train_index, test_index in train_test_split.split(np.zeros((len(latentCode_frame), 3)), labels):
            dataset_train = latentCode_frame.iloc[train_index]
            dataset_test = latentCode_frame.iloc[test_index]
            dataset_train.reset_index(inplace=True)
            dataset_test.reset_index(inplace=True)

            # Built train and test dataset/dataloader
            tr_dataset = Dataset_from_csv(dataset_train, 'GT_label', low_dim_names=low_dim_names)
            tr_dataloader = DataLoader(tr_dataset, batch_size=128, shuffle=True)
            te_dataset = Dataset_from_csv(dataset_test, 'GT_label', low_dim_names=low_dim_names)
            te_dataloader = DataLoader(te_dataset, batch_size=128, shuffle=True)

            # train on train_dataloader
            train_net(model_1, 20, tr_dataloader, num_class=num_class, imbalance_weight_horvath=imbalanced_data)

            # train and test accuracy
            train_acc.append(perf_eval(model_1, tr_dataloader))
            test_acc.append(perf_eval(model_1, te_dataloader))

            perclass_te_acc.append(perf_eval_perclass(model_1, te_dataloader))

        return train_acc, test_acc, perclass_te_acc

    #################################
    ####### Metric two ##############
    #################################
    ## Disregard cluster 7, and only consider strong phenotypic change (>0.5)
    if Metrics[1]:
        non_last_cluster = latentCode_frame['GT_label'] != 7
        strong_phenotype = latentCode_frame['GT_dist_toInit_state'] >= 0.5
        latentCode_frame = latentCode_frame[non_last_cluster & strong_phenotype]
        # For NN, the class must be between 0 - num_class-1
        latentCode_frame['GT_label'] = latentCode_frame['GT_label'].subtract(1)

        ##Make a half / half train test split that have the same percentage of classes as original dataset
        labels = latentCode_frame['GT_label'].values
        train_test_split = StratifiedShuffleSplit(n_splits=num_iteration, test_size=0.5,
                                                  random_state=12)  # Change n_splits if want to have several run

        model_2 = Classifier_Net(zdim=len(low_dim_names)).float().cuda()

        train_acc = []
        test_acc = []
        perclass_te_acc = []
        # For statistical relevance, make several train-test split
        for train_index, test_index in train_test_split.split(np.zeros((len(latentCode_frame), 3)), labels):
            dataset_train = latentCode_frame.iloc[train_index]
            dataset_test = latentCode_frame.iloc[test_index]
            dataset_train.reset_index(inplace=True)
            dataset_test.reset_index(inplace=True)

            # Built train and test dataset/dataloader
            tr_dataset = Dataset_from_csv(dataset_train, 'GT_label', low_dim_names=low_dim_names)
            tr_dataloader = DataLoader(tr_dataset, batch_size=128, shuffle=True)
            te_dataset = Dataset_from_csv(dataset_test, 'GT_label', low_dim_names=low_dim_names)
            te_dataloader = DataLoader(te_dataset, batch_size=128, shuffle=True)

            # train on train_dataloader
            train_net(model_2, 20, tr_dataloader)
            print("finished train")
            # train and test accuracy
            train_acc.append(perf_eval(model_2, tr_dataloader))
            test_acc.append(perf_eval(model_2, te_dataloader))

            perclass_te_acc.append(perf_eval_perclass(model_2, te_dataloader))

        return train_acc, test_acc, perclass_te_acc

    #################################
    ####### Metric three ##############
    #################################
    ## Disregard cluster 7, only strong phenotypic change (>0.5), and METACLUSTER
    # 1&2 vs 3&4 vs 5&6
    if Metrics[2]:
        non_last_cluster = latentCode_frame['GT_label'] != 7
        strong_phenotype = latentCode_frame['GT_dist_toInit_state'] >= 0.5
        latentCode_frame = latentCode_frame[non_last_cluster & strong_phenotype]
        # For NN, the class must be between 0 - num_class-1
        latentCode_frame['GT_label'] = latentCode_frame['GT_label'].subtract(1)

        # Built metacluster (1&2 vs 3&4 vs 5&6)
        latentCode_frame['GT_label'].replace(1, 0, inplace=True)
        latentCode_frame['GT_label'].replace(2, 1, inplace=True)
        latentCode_frame['GT_label'].replace(3, 1, inplace=True)
        latentCode_frame['GT_label'].replace(4, 2, inplace=True)
        latentCode_frame['GT_label'].replace(5, 2, inplace=True)

        ##Make a half / half train test split that have the same percentage of classes as original dataset
        labels = latentCode_frame['GT_label'].values
        train_test_split = StratifiedShuffleSplit(n_splits=num_iteration, test_size=0.5,
                                                  random_state=12)  # Change n_splits if want to have several run

        model_3 = Classifier_Net(zdim=len(low_dim_names), num_of_class=3).float().cuda()

        train_acc = []
        test_acc = []
        perclass_te_acc = []
        # For statistical relevance, make several train-test split
        for train_index, test_index in train_test_split.split(np.zeros((len(latentCode_frame), 3)), labels):
            dataset_train = latentCode_frame.iloc[train_index]
            dataset_test = latentCode_frame.iloc[test_index]
            dataset_train.reset_index(inplace=True)
            dataset_test.reset_index(inplace=True)

            # Built train and test dataset/dataloader
            tr_dataset = Dataset_from_csv(dataset_train, 'GT_label', low_dim_names=low_dim_names)
            tr_dataloader = DataLoader(tr_dataset, batch_size=128, shuffle=True)
            te_dataset = Dataset_from_csv(dataset_test, 'GT_label', low_dim_names=low_dim_names)
            te_dataloader = DataLoader(te_dataset, batch_size=128, shuffle=True)

            # train on train_dataloader
            train_net(model_3, 20, tr_dataloader)

            # train and test accuracy
            train_acc.append(perf_eval(model_3, tr_dataloader))
            test_acc.append(perf_eval(model_3, te_dataloader))

            perclass_te_acc.append(perf_eval_perclass(model_3, te_dataloader))

        return train_acc, test_acc, perclass_te_acc

############################################
##### Old Functions
############################################
# Could be used as hint on how to plot the results 


# def compare_models(list_of_csv,Metrics=[True,False,False],num_iteration=5,num_class=6,class_to_ignore=7,imbalanced_data=False):
#     '''
#     Plot a bar-plot with test accuracy of different projection, stored in several csv files.
#     '''
#     model_names=[]
#     all_means=[]
#     all_stds=[]
#
#     for i, csv_file in enumerate(list_of_csv):
#         train_acc, test_acc, perclass_te_acc = classifier_performance(path_to_csv=csv_file,Metrics=Metrics,num_iteration=num_iteration)
#
#         #In future, probably don-t care about train accuracy
#         all_means.append(np.mean(test_acc))
#         all_stds.append(np.std(test_acc))
#         model_names.append(f'Model {i+1}')
#
#
#         # perclass_means = np.mean(np.array(perclass_te_acc),axis=0)
#         # perclass_stds = np.std(np.array(perclass_te_acc),axis=0)
#         #
#         # #One plot per model for the per-class accuracy
#         # names = [f'cluster {j}' for j in range(1,len(perclass_means)+1)]
#         # fig, ax = plt.subplots()
#         # ax.bar(np.arange(len(names)),perclass_means,yerr=perclass_stds,align='center',alpha=0.5, ecolor='black',capsize=10)
#         # ax.set_ylabel('Test classification accuracy [%]')
#         # ax.set_xticks(np.arange(len(names)))
#         # ax.set_xticklabels(names)
#         # ax.set_title(f'Per Class accuracy of model {i+1}, avg over 10 runs')
#         # ax.yaxis.grid(True)
#         #
#         # plt.tight_layout()
#         # #plt.savefig()
#         # plt.show()
#
#     fig2, ax2 = plt.subplots(figsize=(6,6))
#     ax2.bar(np.arange(len(model_names)),all_means,yerr=all_stds,align='center',alpha=0.5, ecolor='black',capsize=10)
#     for i, pos in enumerate(all_means):
#         ax2.text(i, pos-5,f'{np.round(pos,2)} +- {np.round(all_stds[i],2)}',ha='center',color='black',fontweight='bold')
#     ax2.set_ylabel('Classification accuracy [%]')
#     ax2.set_xticks(np.arange(len(model_names)))
#     ax2.set_xticklabels(model_names)
#     ax2.set_title('Classifier accuracy (avg over 10 runs) - Metric 3')
#     ax2.yaxis.grid(True)
#
#     plt.tight_layout()
#     #plt.savefig()
#     plt.show()


# model1 = 'DataSets/Model1_Good_metadata.csv'
# model2 = 'DataSets/Model2_SmallFail_metadata.csv'
# model3 = 'DataSets/Model3_BigFail_metadata.csv'
# list_models =[model1,model2,model3]
# compare_models(list_models,Metrics=[False,False,True],num_iteration=10)
