# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-05-29T09:52:18+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-31T11:18:07+10:00

'''
MINE (Mutual Information Neural Estimation) implementation.
Use of a NN to estimate the mutual information between the input space
and the latent space, to evaluate and compare different latent representation.
The NN parametrize f, in the f-divergence representation of mutual information.
Finding the supremum of the expression lead to a lower bound of MI.
'''

import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from torch import cuda

from util.Process_benchmarkDataset import get_dsprites_inference_loader
from util.data_processing import get_inference_dataset
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from tqdm import tqdm
import skimage.measure
from scipy.stats import entropy

device = torch.device('cpu' if not cuda.is_available() else 'cuda')


##################################################################
##### Critic ( f(x,y) ) and baseline ( a(y) ) Neural Networks
##################################################################

### This is the Google Implementation of estimating a variatonal bound on MI with NN
### To obtain great number of sample drawn from the marginal (that lead to more stable results)
### -> compute a Batch Size * Batch Size Matrix where each element i,j are NN(Xi,proj_j)
### -> For computational burden limitation we opted for a Separable critic
### -> Still accurate for InfoNCE loss and Interpolate alpha loss that we'll use
class MINE(nn.Module):
    def __init__(self, input_dim, zdim=3):
        super(MINE, self).__init__()

        self.input_dim = input_dim
        self.zdim = zdim
        self.moving_average = None

        self.MLP_g = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
        )
        self.MLP_h = nn.Sequential(
            nn.Linear(zdim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )

    def forward(self, x, z):
        x = x.view(-1, self.input_dim)
        z = z.view(-1, self.zdim)
        x_g = self.MLP_g(x)  # Batchsize x 32
        y_h = self.MLP_h(z)  # Batchsize x 32
        scores = torch.matmul(y_h, torch.transpose(x_g, 0, 1))

        return scores  # Each element i,j is a scalar in R. f(xi,proj_j)


# Small MLP to compute the baseline
class baseline_MLP(nn.Module):
    def __init__(self, input_dim):
        super(baseline_MLP, self).__init__()

        self.input_dim = input_dim

        self.MLP = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        res = self.MLP(x)  # Batchsize x 32

        # Output a scalar which is the log-baseline : log a(y)  for interpolated bound
        return res


#####################################
##### infoNCE Lower Bound ###########
#####################################

# Compute the Noise Constrastive Estimation (NCE) loss
def infoNCE_bound(scores):
    '''Bound from Van Den Oord and al. (2018)'''
    nll = torch.mean(torch.diag(scores) - torch.logsumexp(scores, dim=1))
    k = scores.size()[0]
    mi = np.log(k) + nll

    return mi


#####################################
##### NWJ Lower Bound ###############
#####################################

def tuba_lower_bound(scores, log_baseline=None):
    if log_baseline is not None:
        scores -= log_baseline[:, None]
    batch_size = torch.tensor(scores.size()[0], dtype=torch.float32)
    joint_term = torch.mean(torch.diag(scores))
    marg_term = torch.exp(reduce_logmeanexp_nodiag(scores))
    return 1. + joint_term - marg_term


def nwj_bound(scores):
    return tuba_lower_bound(scores - 1.)


#####################################
##### interpolated Lower Bound ######
#####################################

# Compute interporlate lower bound of MI
def interp_bound(scores, baseline, alpha_logit):
    '''
    New lower bound on mutual information proposed by Ben Poole and al.
    in "On Variational Bounds of Mutual Information"
    It allows to explictily control the biais-variance trade-off.
    For MI estimation -> This bound with a small alpha is much more stable but
    still small biais than NWJ / Mine-f bound !
    Return a scalar, the lower bound on MI
    '''
    batch_size = scores.size()[0]
    nce_baseline = compute_log_loomean(scores)

    interpolated_baseline = log_interpolate(nce_baseline,
                                            baseline[:, None].repeat(1, batch_size),
                                            alpha_logit)  # Interpolate NCE baseline with a learnt baseline

    # Marginal distribution term
    critic_marg = scores - torch.diag(interpolated_baseline)[:, None]  # None is equivalent to newaxis
    marg_term = torch.exp(reduce_logmeanexp_nodiag(critic_marg))

    # Joint distribution term
    critic_joint = torch.diag(scores)[:, None] - interpolated_baseline
    joint_term = (torch.sum(critic_joint) - torch.sum(torch.diag(critic_joint))) / (batch_size * (batch_size - 1.))
    return 1 + joint_term - marg_term


def log_interpolate(log_a, log_b, alpha_logit):
    '''Numerically stable implmentation of log(alpha * a + (1-alpha) *b)
    Compute the log baseline for the interpolated bound
    baseline is a(y)'''
    log_alpha = -F.softplus(torch.tensor(-alpha_logit).to(device))
    log_1_minus_alpha = -F.softplus(torch.tensor(alpha_logit).to(device))
    y = torch.logsumexp(torch.stack((log_alpha + log_a, log_1_minus_alpha + log_b)), dim=0)
    return y


def compute_log_loomean(scores):
    '''Compute the log leave one out mean of the exponentiated scores'''
    max_scores, _ = torch.max(scores, dim=1, keepdim=True)
    lse_minus_max = torch.logsumexp(scores - max_scores, dim=1, keepdim=True)
    d = lse_minus_max + (max_scores - scores)
    d_not_ok = torch.eq(d, 0.)
    d_ok = ~d_not_ok
    safe_d = torch.where(d_ok, d, torch.ones_like(d).to(device))  # Replace zeros by 1 in d

    loo_lse = scores + (safe_d + torch.log(-torch.expm1(-safe_d)))  # Stable implementation of sotfplus_inverse
    loo_lme = loo_lse - np.log(scores.size()[1] - 1.)
    return loo_lme


def reduce_logmeanexp_nodiag(x, axis=None):
    batch_size = x.size()[0]
    logsumexp = torch.logsumexp(x - torch.diag(np.inf * torch.ones(batch_size).to(device)), dim=[0, 1])
    num_elem = batch_size * (batch_size - 1.)
    return logsumexp - torch.log(torch.tensor(num_elem).to(device))

def discretised_entropy_latent(latent, bin_size=10000):
    """
    :param latent: batch * 3
    :param bin_size: num of intervals ### select large bin_size as a closer approach to the true entropy https://doi.org/10.1063/1.4995123
    :return: the average entropy of discretised latent space
    """
    bins = np.linspace(np.min(latent), np.max(latent), bin_size)
    digitized = np.digitize(np.array(latent), bins) # Return the indices of the bins to which each value belongs to.
    _, counts = np.unique(digitized, return_counts=True) # compute the coun
    return entropy(counts, base=2)

#####################################
##### Train Critic and Baseline #####
#####################################
def train_MINE(MINE, path_to_csv, low_dim_names, epochs, infer_dataloader, bound_type='infoNCE', baseline=None,
               alpha_logit=0.):
    '''
    Need a CSV that link every input (single cell images) to its latent code
    infer_dataloader loads file_name unique ID alongside the batch, that enables to
    retrive latent code of each sample from the csv file
    Use the function get_inference_dataset from data_processing.py for that purpose

    Params :
        MINE (nn.Module) : MINE network to train
        path_to_csv (string or DataFrame) : Path to a csv file or directly a DataFrame that contains latent codes as well as unique ID that link to the single cell images (raw data)
        low_dim_names ([string]) : Names of the columns that stores the latent codes
        inter_dataloader (DataLoader) : Use the function get_inference_dataset from data_processing.py to get the right DataLoader
        bound_type (string) : argmunent defines the type of lower bound on MI that is used ('infoNCE', 'NWJ' or 'interpolated')
        baseline (None or nn.Module) : If trainable baseline is used, give the NN to train. (Guideline : only the case for interpolated bound)
        alpha_logit (float) : Weight of the interpolated bound

    Return the history of the training procedure
    '''

    ######################################
    ###### Parameter to bias-variance trade-off
    #######################################
    alpha_logit = alpha_logit

    if isinstance(path_to_csv, str):
        Metadata_csv = pd.read_csv(path_to_csv)
    else:
        Metadata_csv = path_to_csv

    Metadata_csv['Unique_ID'] = Metadata_csv['Unique_ID'].astype(str)

    optimizer = optim.Adam(MINE.parameters(), lr=0.001)
    if bound_type == 'interpolated':
        assert baseline != None, "please provide a valid NN to represent the baseline a(y)"
        optimizer = optim.Adam(list(MINE.parameters()) + list(baseline.parameters()), lr=0.001)

    decayRate = 0.2
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=80, gamma=decayRate)

    history_MI = []

    for epoch in range(epochs):
        miss_cell_counter = 0
        MI_epoch = 0
        for i, (data, labels, file_names) in enumerate(infer_dataloader):

            list_ids = [file_name for file_name in file_names]

            # Retrieve the corresponding latent code in csv_file
            # reindex return nan if corresponding ID doesn't exist in projection
            batch_info = Metadata_csv.set_index('Unique_ID').reindex(list_ids).reset_index(inplace=False)
            # Manage the possibility of missing projection code (happens with UMAP sometimes)
            if batch_info[low_dim_names[0]].isna().any():  # At least 1 projection code is missing
                inds = np.where(batch_info[low_dim_names[0]].isna())[0]
                miss_cell_counter += len(inds)
                # Supress those data point in both low and high dim data
                batch_info.dropna(subset=[low_dim_names[0]], inplace=True)
                data = data[[not (i in inds) for i in np.arange(data.size(0))]]

            batch_latentCode = np.array([list(code) for code in zip(batch_info[low_dim_names[0]], batch_info[low_dim_names[1]],
                                                           batch_info[low_dim_names[2]])])

            entropy_latent_batch = discretised_entropy_latent(batch_latentCode)

            ### compute the entropy for individual images before copy to GPU
            # convert to data to gray scale first
            gray_scale_data = torch.mean(data, axis=1)
            entropy_img_batch = skimage.measure.shannon_entropy(gray_scale_data)

            batch_latentCode = torch.from_numpy(batch_latentCode).float().to(device)
            data = data.to(device)
            MI_loss = None
            if bound_type == 'infoNCE':  # Constant Baseline
                scores = MINE(data, batch_latentCode)
                MI_xz = infoNCE_bound(scores)
                MI_loss = -MI_xz
            elif bound_type == 'NWJ':  # Constant Baseline
                scores = MINE(data, batch_latentCode)
                MI_xz = nwj_bound(scores)
                MI_loss = -MI_xz
            elif bound_type == 'interpolated':  # Learnt Baseline
                scores = MINE(data, batch_latentCode)
                log_baseline = torch.squeeze(baseline(batch_latentCode))
                alpha_logit = alpha_logit  # sigmoid(-5) = 0.01, that correspond to an alpha of 0.01
                MI_xz = interp_bound(scores, log_baseline, alpha_logit)
                MI_loss = -MI_xz
            else:
                assert False, "Please give a valid bound_type, 'infoNCE', 'NWJ' or 'interpolated'"

            optimizer.zero_grad()
            MI_loss.backward()
            optimizer.step()

            MI_epoch += np.divide(2 * MI_xz.detach().cpu().numpy(), entropy_img_batch + entropy_latent_batch)

            if i % 2 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMI: {:.4f} \t Image entropy: {:.4f} \t Latent entropy: {:.4f}'.format(
                    epoch, i * len(data), len(infer_dataloader.dataset),
                           100. * i / len(infer_dataloader),
                    MI_xz.item(), entropy_img_batch, entropy_latent_batch), end='\r')

        MI_epoch /= len(infer_dataloader)
        history_MI.append(MI_epoch)
        # lr_scheduler.step()

        if epoch == 0: print(f'{miss_cell_counter} single cells were not find in the data projection!!!')

    print('Finished Training')
    return np.asarray(history_MI)


def compute_MI(data_csv, low_dim_names=['x_coord', 'y_coord', 'z_coord'], path_to_raw_data='DataSets/Synthetic_Data_1',
               save_path=None, batch_size=512, alpha_logit=-5., bound_type='infoNCE', epochs=300, logger=None, feature_size=64*64*3):
    '''Compute MI (MINE framework) between input data and latent representation.
    Projection coordinates need to be store in the csv file under the columns 'low_dim_names'
    Raw data (image) are loaded by batch from 'path_to_raw_data'

    Save MI history as pkl, and a png plot of the MINE training.
    Return the MI final value (convergence value, mean of 50 last epochs)

    bound_type = 'infoNCE', 'NWJ' or 'interpolated'
        define the bound on MI that is used to estimate and optimize MI
        -'infoNCE' van den Oord implementation (Noise Contrastive Estimation loss)
        -'NWJ' = Mine-f bound
        -'interpolated' Ben Poole implementation, with alpha = 0.01
        Guideline : Use NCE for representation learning, interpolated for MI estimation
    '''

    batch_size = batch_size
    input_size = 64  # CHANGE DEPENDING THE DATASET ############
    epochs = epochs

    if feature_size == 64 * 64:
        _, infer_dataloader = get_dsprites_inference_loader(batch_size=512, shuffle=True)
    else:
        _, infer_dataloader = get_inference_dataset(path_to_raw_data, batch_size, input_size, shuffle=True, droplast=True)

    MINEnet = MINE(feature_size, zdim=len(low_dim_names))  # CHANGE DEPENDING ON DATASET ###########
    MINEnet.to(device)

    baseline = None
    if bound_type == 'interpolated':
        baseline = baseline_MLP(3)  # a(y), take y as input
        baseline.to(device)

    MI_history = train_MINE(MINEnet, data_csv, low_dim_names, epochs, infer_dataloader, bound_type, baseline,
                            alpha_logit)

    if save_path != None:
        MI_pkl_path = save_path + f'/{len(low_dim_names)}_MI_training_history.pkl'
        with open(MI_pkl_path, 'wb') as f:
            pkl.dump(MI_history, f, protocol=pkl.HIGHEST_PROTOCOL)

    MI_Score = np.mean(MI_history[-50:])

    fig, ax = plt.subplots()
    ax.plot(MI_history)
    ax.axhline(y=MI_Score, color='r', ls='--', label=str(np.round(MI_Score, 2)))
    ax.legend()
    ax.set_xlabel('num epochs')
    ax.set_ylabel('Mutual Information')
    ax.set_title(f"Mutual information estimation with bound '{bound_type}'")
    if save_path != None:
        plt.savefig(save_path + f'/{len(low_dim_names)}_MI_score_plot.png')
    if logger: logger.experiment.add_figure(tag="MI metriecs", figure=fig, close=True)

    return MI_Score
