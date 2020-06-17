# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-05-29T09:52:18+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-06-12T14:30:35+10:00

'''
MINE (Mutual Information Neural Estimation) implementation.
Use of a NN to estimate the mitual information between the input space
and the latent space, to evaluate and compare different latent representation.
The NN parametrize f, in the f-divergence representation of mutual information.
Finding the supremum of the expression lead to a lower bound of MI.
'''

import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
import torch.optim as optim
import pandas as pd
import numpy as np

class MINE(nn.Module):
    def __init__(self,input_dim,zdim=3):
        super(MINE, self).__init__()

        self.input_dim = input_dim
        self.zdim = zdim
        self.moving_average = None

        self.MLP = nn.Sequential(
            nn.Linear(input_dim + zdim, 1000),
            nn.LeakyReLU(),
            #nn.Linear(2000, 1000),
            #nn.LeakyReLU(),
            nn.Linear(1000, 100),
            nn.LeakyReLU(),
            #nn.Linear(100, 10),
            #nn.LeakyReLU(),
            nn.Linear(100, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_normal_(m.weight,gain=math.sqrt(2./(1+0.01))) #Gain adapted for LeakyReLU activation function
                m.bias.data.fill_(0.01)

    def forward(self, x, z):
        x = x.view(-1,self.input_dim)
        z = z.view(-1,self.zdim)
        x = torch.cat((x,z),1) #Simple concatenation of x and z
        value = self.MLP(x).squeeze()

        return value #Output is a scalar in R (MI estimate at convergence)


def permute_dims(z):
    '''Permutation in only a trick to be able to get sample from the marginal
    q(z) in a simple manner, to evaluate the variational representation of f-divergence
    '''
    assert z.dim() == 2 #In the form of  batch x latentVariables
    B, _ = z.size()
    perm = torch.randperm(B).cuda()
    perm_z = z[perm]
    return perm_z



def train_MINE(MINE,path_to_csv,epochs,infer_dataloader,train_GPU):
    '''
    Need a CSV that link every input (single cell images) to its latent code
    trainloader load file_name unique ID alongside the batch, that enables to
    retrive latent code of each sample from the csv file
    Use the function get_inference_dataset from data_processing.py for that purpose
    '''

    Metadata_csv = pd.read_csv(path_to_csv)

    optimizer = optim.Adam(MINE.parameters(),lr=0.0001)
    decayRate = 0.6
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=40, gamma=decayRate)

    history_MI = []
    ma_rate = 0.001 #Moving average rate (unbiased trick)

    for epoch in range(epochs):

        MI_epoch = 0

        for i, (data, labels, file_names) in enumerate(infer_dataloader):

            list_ids = [file_name for file_name in file_names]
            if train_GPU:
                # make sure this lives on the GPU
                data = data.cuda()

            #Retrieve the corresponding latent code in csv_file

            batch_info = Metadata_csv.set_index('Unique_ID').loc[list_ids].reset_index(inplace=False)
            batch_latentCode = [list(code) for code in zip(batch_info.x_coord,batch_info.y_coord,batch_info.z_coord)]
            batch_latentCode = torch.from_numpy(np.array(batch_latentCode)).float().cuda()
            t_xz = MINE(data,batch_latentCode) # from joint distribution
            t_xz_tilda = MINE(data,permute_dims(batch_latentCode)) # from product of marginal distribution

            #Estimation of the Mutual Info between X and Z
            #We use the Donsker-Varadhan represensation of MI because if this the tightest
            #lower bound on MI !
            #Also we you an unbiased form with moving average as suggested in original MINE paper
            et = torch.mean(torch.exp(t_xz_tilda))
            if MINE.moving_average is None:
                MINE.moving_average = et.detach().item()
            MINE.moving_average += ma_rate * (et.detach().item() - MINE.moving_average)

            #MI_xz = (t_xz.mean() - (torch.exp(t_xz_tilda -1).mean())) #f-divergence representation
            MI_xz = torch.mean(t_xz) - torch.log(et) #* et.detach() / MINE.moving_average

            #We want to maximize this output -> minimize its negative
            MI_loss = -MI_xz

            optimizer.zero_grad()
            MI_loss.backward()
            optimizer.step()

            MI_epoch += MI_xz


            if i % 2 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMI: {:.6f}'.format(
                    epoch, i * len(data), len(infer_dataloader.dataset),
                           100. * i / len(infer_dataloader),
                           MI_xz.item() ),end='\r')

        MI_epoch /= len(infer_dataloader)
        history_MI.append(MI_epoch.detach().cpu().numpy())
        lr_scheduler.step()

        if epoch % 1 == 0:
            print('==========> Epoch: {} ==========> MI: {:.4f} with lr : {:.8f}'.format(epoch, MI_epoch, lr_scheduler.get_last_lr()[0]))

    print('Finished Training')
    return history_MI
