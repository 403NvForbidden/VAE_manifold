# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-05-25T11:37:48+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-05-25T18:37:56+10:00

'''
File containing different metrics that are used to evaluate the
latent space quality
'''

import pandas as pd
import numpy as np
from classifier_net import Dataset_from_csv, Classifier_Net, train_net, perf_eval, perf_eval_perclass
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt


def classifier_performance(path_to_csv,Metrics=[True,False,False],num_iteration=5):
    '''
    Given a CSV-file containing a 3D latent code to evaluate, built a simple
    (200 unit single hidden layer) NN classifier. Test accuracy can be used as
    a numerical value to assess the quality of the latent code.
    High accuracy -> the latent dimensions present the data in a way where ground
    truth cluster can easily be discriminate

    Metric 1 : Test classification performance on all single cell except uniform cluster (7)
    Metric 2 : Same, but only strong phenotype change (>0.5)
    Metric 3 : Same, but meta-cluster. Discriminate between 1&2, 3&4 and 5&6
    '''
    #################################
    ####### Metric One ##############
    #################################
    ## All cells included, except cluster 7
    if Metrics[0]:
        latentCode_frame = pd.read_csv(path_to_csv)
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
            tr_dataset = Dataset_from_csv(dataset_train,'GT_label')
            tr_dataloader = DataLoader(tr_dataset,batch_size=128,shuffle=True)
            te_dataset = Dataset_from_csv(dataset_test,'GT_label')
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
        latentCode_frame = pd.read_csv(path_to_csv)
        non_last_cluster = latentCode_frame['GT_label'] != 7
        strong_phenotype = latentCode_frame['GT_dist_toMax_phenotype']>=0.5
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
            tr_dataset = Dataset_from_csv(dataset_train,'GT_label')
            tr_dataloader = DataLoader(tr_dataset,batch_size=128,shuffle=True)
            te_dataset = Dataset_from_csv(dataset_test,'GT_label')
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
        latentCode_frame = pd.read_csv(path_to_csv)
        non_last_cluster = latentCode_frame['GT_label'] != 7
        strong_phenotype = latentCode_frame['GT_dist_toMax_phenotype']>=0.5
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
            tr_dataset = Dataset_from_csv(dataset_train,'GT_label')
            tr_dataloader = DataLoader(tr_dataset,batch_size=128,shuffle=True)
            te_dataset = Dataset_from_csv(dataset_test,'GT_label')
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

        #In future, probably don-t care about test accuracy
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

    fig2, ax2 = plt.subplots()
    ax2.bar(np.arange(len(model_names)),all_means,yerr=all_stds,align='center',alpha=0.5, ecolor='black',capsize=10)
    ax2.set_ylabel('Classification accuracy [%]')
    ax2.set_xticks(np.arange(len(model_names)))
    ax2.set_xticklabels(model_names)
    ax2.set_title('NN Classifier acc (avg over 10 runs) as latent code performance metric')
    ax2.yaxis.grid(True)

    plt.tight_layout()
    #plt.savefig()
    plt.show()

#Load the appropriate CSV file
name_of_csv1 = 'DataSets/Sacha_Metadata_3dlatentVAE_20200523.csv'
name_of_csv2 = 'DataSets/Sacha_Metadata_3dlatentVAEFAIL_20200524.csv'
name_of_csv3 = 'DataSets/Sacha_Metadata_3dlatentVAEbigFAIL_20200525.csv'

compare_models([name_of_csv1,name_of_csv2,name_of_csv3],Metrics=[False,False,True],num_iteration=10)
