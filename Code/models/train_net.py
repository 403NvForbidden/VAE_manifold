# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-06T12:15:33+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-31T10:26:36+10:00

'''
File containing main function to train the different VAE models with proper single cell images dataset

####### 1 #######
train(), test() and train_VAE_model() :
--> To train VAE with ELBO objective function (Vanilla VAE and SCVAE)

####### 2 #######
train_infoM_epoch(), test_infoM_epoch() and train_InfoMAX_model() :
--> To train InfoMAX VAE, jointly optimizing VAE and MLP MI estimator

####### 3 #######
train_feedback(), train_Simple_VAE() :
--> Train a VAE with human feedback stored in a CSV file
'''

import pandas as pd
import numpy as np
from timeit import default_timer as timer

import torch
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
from torch import cuda
from torchvision.utils import save_image, make_grid

from models.networks import VAE
from util.helpers import plot_latent_space, show, EarlyStopping
from models.infoMAX_VAE import infoNCE_bound


def kl_divergence(mu, logvar):
    kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
    return kld

def scalar_loss(data, loss_recon, mu_z, logvar_z, beta):
    # calculate scalar loss of VAE1
    loss_recon *= data.size(1) * data.size(2) * data.size(3)
    loss_recon.div(data.size(0))
    loss_kl = kl_divergence(mu_z, logvar_z)
    return loss_recon + beta * loss_kl, loss_kl

###################################################
##### 2 stage VAE training ##############
###################################################
def train_2_stage_VAE_epoch(num_epochs, VAE_1, VAE_2, optimizer_1, optimizer_2, train_loader, device='cuda'):
    '''
        Train a VAE model with standard ELBO objective function for one single epoch

        Params :
            epoch (int) : the considered epoch
            model (nn.Module) : VAE model to train
            optimizer (optim.Optimizer) : Optimizer used for training
            train_loader (DataLoader) : Dataloader used for training

        Return the average loss, as well as average of the two main terms (reconstruction and KL)
    '''
    # toggle model to train mode
    VAE_1.train()
    VAE_2.train()

    # Store the different loss per iteration (=batch)
    loss_overall_iter = []
    global_VAE_iter_1, kl_loss_iter_1, recon_loss_iter_1 = [], [], []
    global_VAE_iter_2, kl_loss_iter_2, recon_loss_iter_2 = [], [], []

    start = timer()

    criterion_recon = nn.BCEWithLogitsLoss().to(device)  # more stable than handmade sigmoid as last layer and BCELoss

    # each `data` is of BATCH_SIZE samples and has shape [batch_size, 4, 128, 128]
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data).to(device)

        # push whole batch of data through VAE.forward() to get recon_loss
        x_recon_1, mu_z_1, logvar_z_1, _ = VAE_1(data)
        x_recon_2, mu_z_2, logvar_z_2, _ = VAE_2(data)

        # calculate scalar loss of VAE1
        loss_recon_1 = criterion_recon(x_recon_1, data)
        loss_VAE_1, loss_kl_1 = scalar_loss(data, loss_recon_1, mu_z_1, logvar_z_1, VAE_1.beta)
        # calculate scalar loss of VAE2
        loss_recon_2 = criterion_recon(x_recon_2, data)
        loss_VAE_2, loss_kl_2 = scalar_loss(data, loss_recon_2, mu_z_2, logvar_z_2, VAE_2.beta)
        # total loss
        loss_overall = loss_VAE_2 + 0.8 * loss_VAE_1 # as auxiliary loss

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        loss_overall.backward()
        optimizer_1.step()
        optimizer_2.step()

        # record the loss
        loss_overall_iter.append(loss_overall.item())
        global_VAE_iter_1, global_VAE_iter_2 =  global_VAE_iter_1 + [loss_VAE_1.item()], global_VAE_iter_2 + [loss_VAE_2.item()]
        recon_loss_iter_1, recon_loss_iter_2 = recon_loss_iter_1 + [loss_recon_1.item()], global_VAE_iter_2 + [loss_recon_2.item()]
        kl_loss_iter_1, kl_loss_iter_2 = kl_loss_iter_1 + [loss_kl_1.item()], kl_loss_iter_2 + [loss_kl_2.item()]

        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tVAE1 Loss: {:.6f}\tVAE2 Loss: {:.6f}'.format(
                num_epochs, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss_VAE_1.item(), loss_VAE_2.item()), end='\r')

    if (num_epochs % 5 == 0) or (num_epochs == 1):
        print('==========> Epoch: {} ==========> Average loss: {:.4f}'.format(num_epochs, np.mean(loss_overall_iter)))
        print(f'{timer() - start:.2f} seconds elapsed in epoch.')
        print(f'VAE_1 VAE loss: {np.mean(global_VAE_iter_1):.2f}, KL loss : {np.mean(global_VAE_iter_2):.2f}')
        print(f'VAE_1 reconstruction loss: {np.mean(recon_loss_iter_1):.2f}, KL loss : {np.mean(kl_loss_iter_1):.2f}')
        print(f'VAE_2 reconstruction loss: {np.mean(recon_loss_iter_2):.2f}, KL loss : {np.mean(kl_loss_iter_2):.2f}')

    return np.mean(loss_overall_iter), np.mean(kl_loss_iter_1), np.mean(kl_loss_iter_2), np.mean(recon_loss_iter_1), np.mean(recon_loss_iter_2)

def test_2_stage_VAE_epoch(epoch, model, optimizer, test_loader, train_on_gpu=True):
    pass

def train_2_stage_VAE_model(num_epochs, VAE_1, VAE_2, optimizer1, optimizer2, train_loader, valid_loader,
                            save_path='', device='cpu'):
    '''
        Define VAE 1:

        Params :
            epochs (int) : Maximum number of epochs
            model (nn.Module) : VAE model to train
            optimizer (optim.Optimizer) : Optimizer used for training
            train_loader (DataLoader) : Dataloader used for training
            test_loader (DataLoader) : Dataloader used for evaluation
            saving_path (string) : path to the folder to store the best model

        Return a DataFrame containing the training history, as well as the trained model and the best epoch
    '''
    # Number of epochs already trained (if pre trained)
    try:
        print(f'VAE1, VAE2 has been trained for: {VAE_1.epochs}, {VAE_2.epochs} epochs.\n')
    except:
        VAE_1.epochs, VAE_2.epochs = 0, 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()
    best_epoch = 0

    history = []
    early_stopping = EarlyStopping(patience=30, verbose=True, path=save_path)
    lr_schedul_VAE_1 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer1, step_size=40, gamma=0.6)
    lr_schedul_VAE_2 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer2, step_size=40, gamma=0.6)

    for epoch in range(num_epochs):
        global_VAE_loss, kl_loss_1, kl_loss_2, recon_loss_1, recon_loss_2 = train_2_stage_VAE_epoch(epoch, VAE_1, VAE_2,
                                                                                                    optimizer1,
                                                                                                    optimizer2,
                                                                                                    train_loader,
                                                                                                    device)
        # global_VAE_loss_val, kl_loss_val, recon_loss_val = test_2_stage_VAE_epoch(num_epochs, VAE1, VAE2, optimizer, valid_loader, train_on_gpu)

        # early stopping takes the validation loss to check if it has decereased,
        # if so, model is saved, if not for 'patience' time in a row, the training loop is broken
        early_stopping(global_VAE_loss, VAE_1.epochs)
        best_epoch = early_stopping.stop_epoch

        history.append([global_VAE_loss, kl_loss_1, kl_loss_2, recon_loss_1, recon_loss_2])
        VAE_1.epochs, VAE_2.epochs = VAE_1.epochs + 1, VAE_2.epochs + 1

        lr_schedul_VAE_1.step()
        lr_schedul_VAE_2.step()

        if early_stopping.early_stop:
            print(f'#### Early stopping occured. Best model saved is from epoch {early_stopping.stop_epoch}')
            break

    ### SAVE the model
    early_stopping.save_model(VAE_1, VAE_2)

    history = pd.DataFrame(
        history,
        columns=['global_VAE_loss', 'kl_loss_1', 'kl_loss_2', 'recon_loss_1', 'recon_loss_2'])

    # time
    total_time = timer() - overall_start
    print(f'{total_time:.2f} total seconds elapsed. {total_time / (num_epochs):.2f} seconds per epoch.')
    print('######### TRAINING FINISHED ##########')

    # Attach the optimizer
    VAE_1.optimizer = optimizer1
    VAE_2.optimizer = optimizer2

    return VAE_1, VAE_2, history, best_epoch

###################################################
##### 2 stage VAE_withInfoMax ##############
###################################################
def train_2_stage_infoVAE_epoch(num_epochs, VAE_1, VAE_2, optim_VAE1, optim_VAE2, MLP_1, MLP_2, opti_MLP1, opti_MLP2, train_loader, double_embed=False, device='cuda'):
    '''
        Train a VAE model with standard ELBO objective function for one single epoch

        Params :
            epoch (int) : the considered epoch
            model (nn.Module) : VAE model to train
            optimizer (optim.Optimizer) : Optimizer used for training
            train_loader (DataLoader) : Dataloader used for training

        Return the average loss, as well as average of the two main terms (reconstruction and KL)
    '''
    # toggle model to train mode
    VAE_1.train()
    VAE_2.train()

    # Store the different loss per iteration (=batch)
    loss_overall_iter = []
    global_VAE_iter_1, kl_loss_iter_1, recon_loss_iter_1, MI_iter_1, MI_loss_iter_1 = [], [], [], [], []
    global_VAE_iter_2, kl_loss_iter_2, recon_loss_iter_2, MI_iter_2, MI_loss_iter_2 = [], [], [], [], []

    start = timer()

    criterion_recon = nn.BCEWithLogitsLoss().to(device)  # more stable than handmade sigmoid as last layer and BCELoss

    # each `data` is of BATCH_SIZE samples and has shape [batch_size, 4, 128, 128]
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data).to(device)

        # push whole batch of data through VAE.forward() to get recon_loss
        x_recon_1, mu_z_1, logvar_z_1, z_1 = VAE_1(data)
        scores_1 = MLP_1(data, z_1)

        if not double_embed:
            x_recon_2, mu_z_2, logvar_z_2, z_2 = VAE_2(data)
            scores_2 = MLP_2(data, z_2)
        else:
            x_recon_2, mu_z_2, logvar_z_2, z_2 = VAE_2(z_1)
            scores_2 = MLP_2(z_1, z_2)

        # Estimation of the Mutual Info between X and Z
        MI_xz_1 = infoNCE_bound(scores_1)
        MI_xz_2 = infoNCE_bound(scores_2)


        # calculate scalar loss of VAE1
        loss_recon_1 = criterion_recon(x_recon_1, data)
        loss_VAE_1, loss_kl_1 = scalar_loss(data, loss_recon_1, mu_z_1, logvar_z_1, VAE_1.beta)
        # calculate scalar loss of VAE2
        loss_recon_2 = criterion_recon(x_recon_2, data)
        loss_VAE_2, loss_kl_2 = scalar_loss(data, loss_recon_2, mu_z_2, logvar_z_2, VAE_2.beta)
        # total loss
        # loss_overall = loss_VAE_1 + gamma * loss_VAE_2
        loss_overall = (loss_VAE_2 - VAE_2.alpha * MI_xz_2) + 0.8 * (loss_VAE_1 - VAE_1.alpha * MI_xz_1)

        # Step 1 : Optimization of VAE based on the current MI estimation
        optim_VAE1.zero_grad()
        optim_VAE2.zero_grad()
        loss_overall.backward(retain_graph=True) #Important argument, we backpropagated two times over MI_xz)
        optim_VAE1.step()
        optim_VAE2.step()

        # Step 2 : Optimization of the MLP to improve the MI estimation
        opti_MLP1.zero_grad()
        opti_MLP2.zero_grad()
        MI_loss_1 = -MI_xz_1
        MI_loss_2 = -MI_xz_2
        MI_loss_1.backward(retain_graph=True) # Important argument
        MI_loss_2.backward()
        opti_MLP1.step()
        opti_MLP2.step()

        # record the loss
        loss_overall_iter.append(loss_overall.item())
        global_VAE_iter_1, global_VAE_iter_2 = global_VAE_iter_1 + [loss_VAE_1.item()], global_VAE_iter_2 + [loss_VAE_2.item()]
        recon_loss_iter_1, recon_loss_iter_2 = recon_loss_iter_1 + [loss_recon_1.item()], global_VAE_iter_2 + [loss_recon_2.item()]
        kl_loss_iter_1, kl_loss_iter_2 = kl_loss_iter_1 + [loss_kl_1.item()], kl_loss_iter_2 + [loss_kl_2.item()]

        MI_iter_1.append(MI_xz_1.item())
        MI_iter_2.append(MI_xz_2.item())
        MI_loss_iter_1.append(MI_loss_1.item())
        MI_loss_iter_2.append(MI_loss_2.item())

        ### comment out for fast training
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tVAE1 Loss: {:.6f}\tVAE2 Loss: {:.6f}'.format(
                num_epochs, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss_VAE_1.item(), loss_VAE_2.item()), end='\r')

    if (num_epochs % 5 == 0) or (num_epochs == 1):
        print('==========> Epoch: {} ==========> Average loss: {:.4f}'.format(num_epochs, np.mean(loss_overall_iter)))
        print(f'{timer() - start:.2f} seconds elapsed in epoch.')
        print(f'VAE_1 VAE loss: {np.mean(global_VAE_iter_1):.2f}, KL loss : {np.mean(global_VAE_iter_2):.2f}')
        print(f'VAE_1 reconstruction loss: {np.mean(recon_loss_iter_1):.2f}, KL loss : {np.mean(kl_loss_iter_1):.2f} MI : {np.mean(MI_iter_1):.2f}')
        print(f'VAE_2 reconstruction loss: {np.mean(recon_loss_iter_2):.2f}, KL loss : {np.mean(kl_loss_iter_2):.2f} MI : {np.mean(MI_iter_2):.2f}')

    return np.mean(loss_overall_iter), np.mean(kl_loss_iter_1), np.mean(kl_loss_iter_2), np.mean(recon_loss_iter_1), np.mean(recon_loss_iter_2), np.mean(MI_iter_1), np.mean(MI_iter_2), np.mean(MI_loss_iter_1), np.mean(MI_loss_iter_2)

def train_2_stage_infoVAE_model(num_epochs, VAE_1, VAE_2, opti_VAE1, opti_VAE2, MLP_1, MLP_2, opti_MLP1, opti_MLP2, train_loader, valid_loader,
                                save_path='', double_embed=False, device='cuda'):

    # Number of epochs already trained (if pre trained)
    try:
        print(f'VAE1, VAE2 has been trained for: {VAE_1.epochs}, {VAE_2.epochs} epochs.\n')
    except:
        VAE_1.epochs, VAE_2.epochs = 0, 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()
    best_epoch = 0

    history = []
    early_stopping = EarlyStopping(patience=30, verbose=True, path=save_path)
    lr_schedul_VAE_1 = torch.optim.lr_scheduler.StepLR(optimizer=opti_VAE1, step_size=40, gamma=0.6)
    lr_schedul_VAE_2 = torch.optim.lr_scheduler.StepLR(optimizer=opti_VAE2, step_size=40, gamma=0.6)
    lr_schedul_MLP1 = torch.optim.lr_scheduler.StepLR(optimizer=opti_MLP1, step_size=40, gamma=0.6)
    lr_schedul_MLP2 = torch.optim.lr_scheduler.StepLR(optimizer=opti_MLP2, step_size=40, gamma=0.6)



    for epoch in range(num_epochs):
        global_VAE_loss, kl_loss_1, kl_loss_2, recon_loss_1, recon_loss_2, MI_iter_1, MI_iter_2, MI_loss_1, MI_loss_2 = train_2_stage_infoVAE_epoch(epoch, VAE_1, VAE_2,
                                                                                                                                                    opti_VAE1,
                                                                                                                                                    opti_VAE2, MLP_1, MLP_2, opti_MLP1, opti_MLP2,
                                                                                                                                                    train_loader, double_embed,
                                                                                                                                                    device)
        # global_VAE_loss_val, kl_loss_val, recon_loss_val = test_2_stage_VAE_epoch(num_epochs, VAE1, VAE2, optimizer, valid_loader, train_on_gpu)

        # early stopping takes the validation loss to check if it has decereased,
        # if so, model is saved, if not for 'patience' time in a row, the training loop is broken
        early_stopping(global_VAE_loss, VAE_1.epochs)
        best_epoch = early_stopping.stop_epoch

        history.append([global_VAE_loss, kl_loss_1, kl_loss_2, recon_loss_1, recon_loss_2, MI_iter_1, MI_iter_2, MI_loss_1, MI_loss_2])
        VAE_1.epochs, VAE_2.epochs = VAE_1.epochs + 1, VAE_2.epochs + 1

        lr_schedul_VAE_1.step()
        lr_schedul_VAE_2.step()
        lr_schedul_MLP1.step()
        lr_schedul_MLP2.step()

        if early_stopping.early_stop:
            print(f'#### Early stopping occured. Best model saved is from epoch {early_stopping.stop_epoch}')
            break

    ### SAVE the model
    early_stopping.save_model(VAE_1, VAE_2)

    history = pd.DataFrame(
        history,
        columns=['global_VAE_loss', 'kl_loss_1', 'kl_loss_2', 'recon_loss_1', 'recon_loss_2', \
                 'MI_iter_1', 'MI_iter_2', 'MI_loss_1', 'MI_loss_2']
    )

    # time
    total_time = timer() - overall_start
    print(f'{total_time:.2f} total seconds elapsed. {total_time / (num_epochs):.2f} seconds per epoch.')
    print('######### TRAINING FINISHED ##########')

    # Attach the optimizer
    VAE_1.optimizer = opti_VAE1
    VAE_2.optimizer = opti_VAE2
    MLP_1.optimizer = opti_MLP1
    MLP_2.optimizer = opti_MLP2

    return VAE_1, VAE_2, MLP_1, MLP_2, history, best_epoch

###################################################
##### Vanilla VAE and SCVAE training ##############
###################################################
def train(epoch, model, optimizer, train_loader, train_on_gpu=True):
    '''
    Train a VAE model with standard ELBO objective function for one single epoch

    Params :
        epoch (int) : the considered epoch
        model (nn.Module) : VAE model to train
        optimizer (optim.Optimizer) : Optimizer used for training
        train_loader (DataLoader) : Dataloader used for training

    Return the average loss, as well as average of the two main terms (reconstruction and KL)
    '''
    # toggle model to train mode
    model.train()

    #Store the different loss per iteration (=batch)
    global_VAE_iter = []
    kl_loss_iter = []
    recon_loss_iter = []

    start = timer()

    criterion_recon = nn.BCEWithLogitsLoss().cuda() #more stable than handmade sigmoid as last layer and BCELoss

    # each `data` is of BATCH_SIZE samples and has shape [batch_size, 4, 128, 128]
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if train_on_gpu:
            data = data.cuda()

        # push whole batch of data through VAE.forward() to get recon_loss
        x_recon, mu_z, logvar_z, _ = model(data)

        # calculate scalar loss
        loss_recon = criterion_recon(x_recon,data)
        loss_recon *= data.size(1)*data.size(2)*data.size(3)
        loss_recon.div(data.size(0))

        loss_kl = kl_divergence(mu_z,logvar_z)
        loss_VAE = loss_recon + model.beta * loss_kl

        optimizer.zero_grad()
        loss_VAE.backward()
        optimizer.step()

        global_VAE_iter.append(loss_VAE.item())
        recon_loss_iter.append(loss_recon.item())
        kl_loss_iter.append(loss_kl.item())

        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss_VAE.item() ),end='\r')

    if (epoch%10==0) or (epoch == 1):
        print('==========> Epoch: {} ==========> Average loss: {:.4f}'.format(epoch, np.mean(global_VAE_iter)))
        print(f'{timer() - start:.2f} seconds elapsed in epoch.')
        print(f'Reconstruction loss : {np.mean(recon_loss_iter):.2f}, KL loss : {np.mean(kl_loss_iter):.2f}')

    return np.mean(global_VAE_iter), np.mean(kl_loss_iter), np.mean(recon_loss_iter)

def test(epoch, model, optimizer, test_loader, train_on_gpu=True):
    '''
    Evaluate a VAE model on validation dataloader with standard ELBO objective function

    Params :
        epoch (int) : the considered epoch
        model (nn.Module) : VAE model to train
        optimizer (optim.Optimizer) : Optimizer used for training
        test_loader (DataLoader) : Dataloader used for evaluation

    Return the average loss, as well as average of the two main terms (reconstruction and KL)
    '''

    with torch.no_grad():
        model.eval()
        global_VAE_iter = []
        kl_loss_iter = []
        recon_loss_iter = []

        criterion_recon = nn.BCEWithLogitsLoss().cuda() #more stable than handmade sigmoid as last layer and BCELoss

        # each data is of BATCH_SIZE (default 128) samples
        for i, (data, _) in enumerate(test_loader):
            if train_on_gpu:
                data = data.cuda()

            # we're only going to infer, so no autograd at all required
            data = Variable(data, requires_grad=False)
            x_recon, mu_z, logvar_z, _ = model(data)

            loss_recon = criterion_recon(x_recon,data)
            loss_recon *= data.size(1)*data.size(2)*data.size(3)
            loss_recon.div(data.size(0))
            loss_kl = kl_divergence(mu_z,logvar_z)

            loss_VAE = loss_recon + model.beta * loss_kl

            global_VAE_iter.append(loss_VAE.item())
            recon_loss_iter.append(loss_recon.item())
            kl_loss_iter.append(loss_kl.item())

    if (epoch%10==0) or (epoch == 1):
        print('Test Errors for Epoch: {} ----> Average loss: {:.4f}'.format(epoch, np.mean(global_VAE_iter)))
    return np.mean(global_VAE_iter), np.mean(kl_loss_iter), np.mean(recon_loss_iter)


def train_VAE_model(epochs, model, optimizer, train_loader, valid_loader, saving_path='best_model.pth', train_on_gpu=True):
    '''
    Main function to train a VAE model with standard ELBO objective function for a given number of epochs
    Possible to train from scratch or resume a training (simply pass a trained VAE as input)

    A Early spotting with patience of 30 epochs is used. Model are automatically saved
    when the validation loss decreases. Training is stopped when the latter didn't decrease
    for 30 epochs, or the maximum number of epochs is reached.

    Params :
        epochs (int) : Maximum number of epochs
        model (nn.Module) : VAE model to train
        optimizer (optim.Optimizer) : Optimizer used for training
        train_loader (DataLoader) : Dataloader used for training
        test_loader (DataLoader) : Dataloader used for evaluation
        saving_path (string) : path to the folder to store the best model

    Return a pandas DataFrame containing the training history, as well as the trained model and the best epoch
    '''
    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()
    best_epoch = 0

    history = []
    early_stopping = EarlyStopping(patience=30,verbose=True,path=saving_path)
    lr_schedul_VAE = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=40, gamma=0.6)

    for epoch in range(model.epochs+1,model.epochs+epochs+1):
        global_VAE_loss, kl_loss, recon_loss = train(epoch, model,optimizer, train_loader, train_on_gpu)
        global_VAE_loss_val, kl_loss_val, recon_loss_val = test(epoch, model,optimizer,valid_loader, train_on_gpu)

        #early stopping takes the validation loss to check if it has decereased,
        #if so, model is saved, if not for 'patience' time in a row, the training loop is broken
        early_stopping(global_VAE_loss_val,VAE=model,MLP=None)
        best_epoch = early_stopping.stop_epoch

        history.append([global_VAE_loss, kl_loss, recon_loss,global_VAE_loss_val, kl_loss_val, recon_loss_val])
        model.epochs += 1
        lr_schedul_VAE.step()

        if early_stopping.early_stop:
            print(f'#### Early stopping occured. Best model saved is from epoch {early_stopping.stop_epoch}')
            break

    history = pd.DataFrame(
        history,
        columns=['global_VAE_loss', 'kl_loss', 'recon_loss',
        'global_VAE_loss_val','kl_loss_val','recon_loss_val'])

    total_time = timer() - overall_start
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    print('######### TRAINING FINISHED ##########')

    # Attach the optimizer
    model.optimizer = optimizer

    return model, history, best_epoch



###################################################
##### InfoMax VAE training ##############
###################################################
def train_infoM_epoch(epoch, VAE, MLP, opti_VAE, opti_MLP, train_loader, train_on_gpu=False):
    '''
    Train a VAE model with InfoMAX VAE objective function for one single epoch
    A VAE and a MLP that estimate mutual information are jointly optimized

    Params :
        epoch (int) : the considered epoch
        VAE (nn.Module) : VAE model to train
        MLP (nn.Module) : MLP model to train (MI estimator)
        opti_VAE (optim.Optimizer) : Optimizer used for VAE training
        opti_MLP (optim.Optimizer) : Optimizer used for MLP training
        train_loader (DataLoader) : Dataloader used for training

    Return the average global loss, as well as average of the different terms
    '''
    # toggle model to train mode
    VAE.train()

    global_VAE_iter = []
    MI_estimation_iter = []
    MI_estimator_loss_iter = []
    kl_loss_iter = []
    recon_loss_iter = []

    start = timer()

    criterion_recon = nn.BCEWithLogitsLoss().cuda() #more stable than handmade sigmoid as last layer and BCELoss

    # each `data` is of BATCH_SIZE samples and has shape [batch_size, 4, H, W]
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if train_on_gpu:
            data = data.cuda()

        #data feed to CNN-VAE
        x_recon, mu_z, logvar_z, z = VAE(data)
        scores = MLP(data, z)

        #Estimation of the Mutual Info between X and Z
        MI_xz = infoNCE_bound(scores)

        loss_recon = criterion_recon(x_recon, data)
        loss_recon *= data.size(1)*data.size(2)*data.size(3)
        loss_recon.div(data.size(0))
        loss_kl = kl_divergence(mu_z, logvar_z)

        loss_VAE = loss_recon + VAE.beta * loss_kl - VAE.alpha * MI_xz

        # Step 1 : Optimization of VAE based on the current MI estimation
        opti_VAE.zero_grad()
        loss_VAE.backward(retain_graph=True) #Important argument, we backpropagated two times over MI_xz
        opti_VAE.step()

        MI_loss = -MI_xz
        # Step 2 : Optimization of the MLP to improve the MI estimation
        opti_MLP.zero_grad()
        MI_loss.backward()
        opti_MLP.step()

        global_VAE_iter.append(loss_VAE.item())
        recon_loss_iter.append(loss_recon.item())
        kl_loss_iter.append(loss_kl.item())
        MI_estimation_iter.append(MI_xz.item())
        MI_estimator_loss_iter.append(MI_loss.item())

        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss_VAE.item() ),end='\r')

    if (epoch%10==0) or (epoch == 1):
        print('==========> Epoch: {} ==========> Average loss: {:.4f}'.format(epoch, np.mean(global_VAE_iter)))
        print(f'{timer() - start:.2f} seconds elapsed in epoch.')
        print(f'Reconstruction loss : {np.mean(recon_loss_iter):.2f}, KL loss : {np.mean(kl_loss_iter):.2f} \n MI : {np.mean(MI_estimation_iter):.2f} ')

    return np.mean(global_VAE_iter), np.mean(MI_estimation_iter), np.mean(MI_estimator_loss_iter), np.mean(kl_loss_iter), np.mean(recon_loss_iter)


def test_infoM_epoch(epoch, VAE, MLP, opti_VAE, opti_MLP, test_loader, train_on_gpu=False):
    '''
    Evaluate a VAE model with InfoMAX VAE objective function for one single epoch

    Params :
        epoch (int) : the considered epoch
        VAE (nn.Module) : VAE model to train
        MLP (nn.Module) : MLP model to train (MI estimator)
        opti_VAE (optim.Optimizer) : Optimizer used for VAE training
        opti_MLP (optim.Optimizer) : Optimizer used for MLP training
        test_loader (DataLoader) : Dataloader used for evaluation

    Return the average global loss, as well as average of the different terms
    '''

    with torch.no_grad():
        VAE.eval()
        global_VAE_iter = []
        MI_estimation_iter = []
        MI_estimator_loss_iter = []
        kl_loss_iter = []
        recon_loss_iter = []

        criterion_recon = nn.BCEWithLogitsLoss().cuda() #more stable than handmade sigmoid as last layer and BCELoss


        # each data is of BATCH_SIZE (default 128) samples
        for batch_idx, (data, _) in enumerate(test_loader):
            if train_on_gpu:
                # make sure this lives on the GPU
                data = data.cuda()

            # we're only going to infer, so no autograd at all required
            data = Variable(data, requires_grad=False)

            #data feed to CNN-VAE
            x_recon, mu_z, logvar_z, z = VAE(data)
            scores = MLP(data,z)
            MI_xz = infoNCE_bound(scores)

            #Estimation of the Mutual Info between X and Z
            MI_loss = -MI_xz

            loss_recon = criterion_recon(x_recon,data)
            loss_recon *= data.size(1)*data.size(2)*data.size(3)
            loss_recon.div(data.size(0))
            loss_kl = kl_divergence(mu_z,logvar_z)

            loss_VAE = loss_recon + VAE.beta * loss_kl - VAE.alpha * MI_xz

            global_VAE_iter.append(loss_VAE.item())
            recon_loss_iter.append(loss_recon.item())
            kl_loss_iter.append(loss_kl.item())
            MI_estimation_iter.append(MI_xz.item())
            MI_estimator_loss_iter.append(MI_loss.item())

    if (epoch%10==0) or (epoch == 1):
        print('Test Errors for Epoch: {} ----> Average loss: {:.4f}'.format(epoch, np.mean(global_VAE_iter)))
    return np.mean(global_VAE_iter), np.mean(MI_estimation_iter), np.mean(MI_estimator_loss_iter), np.mean(kl_loss_iter), np.mean(recon_loss_iter)


def train_InfoMAX_model(epochs,VAE, MLP, opti_VAE, opti_MLP, train_loader, valid_loader, saving_path='best_model.pth', train_on_gpu=False):
    '''
    Main function to train a VAE model with InfoMAX VAE objective function for a given number of epochs
    Standard ELBO objective function with an additional term maximizing mutual information is
    optimized jointly with a MLP that estimate mutual information
    Possible to train from scratch or resume a training (simply pass a trained VAE as input)

    A Early spotting with patience of 30 epochs is used. Model are automatically saved
    when the validation loss decreases. Training is stopped when the latter didn't decrease
    for 30 epochs, or the maximum number of epochs is reached.

    Params :
        epochs (int) : Maximum number of epochs
        VAE (nn.Module) : VAE model to train
        MLP (nn.Module) : MLP model to train (MI estimator)
        opti_VAE (optim.Optimizer) : Optimizer used for VAE training
        opti_MLP (optim.Optimizer) : Optimizer used for MLP training
        train_loader (DataLoader) : Dataloader used for training
        test_loader (DataLoader) : Dataloader used for evaluation
        saving_path (string) : path to the folder to store the best model

    Return a pandas DataFrame containing the training history, as well as the trained models and the best epoch
    '''

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {VAE.epochs} epochs.\n')
    except:
        VAE.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()
    best_epoch = 0

    history = []
    early_stopping = EarlyStopping(patience=30, verbose=True, path=saving_path)
    lr_schedul_VAE = torch.optim.lr_scheduler.StepLR(optimizer=opti_VAE, step_size=40, gamma=0.6)
    lr_schedul_MLP = torch.optim.lr_scheduler.StepLR(optimizer=opti_MLP, step_size=40, gamma=0.6)

    for epoch in range(VAE.epochs+1,VAE.epochs+epochs+1):
        global_VAE_loss, MI_estimation, MI_estimator_loss, kl_loss, recon_loss = train_infoM_epoch(epoch, VAE, MLP, opti_VAE, opti_MLP, train_loader, train_on_gpu)
        global_VAE_loss_val, MI_estimation_val, MI_estimator_loss_val, kl_loss_val, recon_loss_val = test_infoM_epoch(epoch, VAE, MLP, opti_VAE, opti_MLP, valid_loader, train_on_gpu)

        #ealy stopping takes the validation loss to check if it has decereased,
        #if so, model is saved, if not for 'patience' time in a row, the training loop is broken
        early_stopping(global_VAE_loss_val, VAE, MLP)
        best_epoch = early_stopping.stop_epoch

        history.append([global_VAE_loss, MI_estimation, MI_estimator_loss, kl_loss, recon_loss,global_VAE_loss_val, MI_estimation_val, MI_estimator_loss_val, kl_loss_val, recon_loss_val])
        VAE.epochs += 1
        lr_schedul_VAE.step()
        lr_schedul_MLP.step()

        if early_stopping.early_stop:
            print(f'#### Early stopping occured. Best model saved is from epoch {early_stopping.stop_epoch}')
            break

    history = pd.DataFrame(
        history,
        columns=['global_VAE_loss', 'MI_estimation', 'MI_estimator_loss', 'kl_loss', 'recon_loss',
        'global_VAE_loss_val','MI_estimation_val','MI_estimator_loss_val','kl_loss_val','recon_loss_val'])

    total_time = timer() - overall_start
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    print('######### TRAINING FINISHED ##########')

    # Attach the optimizer
    VAE.optimizer = opti_VAE
    MLP.optimizer = opti_MLP

    return VAE, MLP, history, best_epoch



###################################################
##### Train vanilla VAE with Human Guidance #######
###################################################


def train_feedback(epoch, model, optimizer, train_loader, train_on_gpu=True):
    '''
    Train a VAE model with standard EBLO function for one single epoch
    An additional term is present is the objective, to force some points to a defined
    anchors with a MSE loss.
    The idendity of anchored points and the anchors are specified in a CSV file
    that is used by a special DataLoader.
    Please refer to human_guidance/feedback_helpers.py and class 'DSpritesDataset' for more info

    Params :
        epoch (int) : the considered epoch
        model (nn.Module) : VAE model to train
        optimizer (optim.Optimizer) : Optimizer used for VAE training
        train_loader (DataLoader) : Dataloader used for training. Need to be a custom dataloader built to
                take in account feedback from a csv file. Please refer to human_guidance/feedback_helpers.py
                and class 'DSpritesDataset' for more info

    Return the average global loss, as well as average of the different terms
    '''
    # toggle model to train mode
    model.train()

    #Score the different loss per iteration (=batch)
    global_VAE_iter = []
    kl_loss_iter = []
    recon_loss_iter = []

    start = timer()

    criterion_recon = nn.BCEWithLogitsLoss().cuda() #more stable than handmade sigmoid as last layer and BCELoss
    MSE = nn.MSELoss(reduce=False)
    def weighted_mse_loss(input, target, weight):
        return torch.sum(weight * torch.sum(MSE(input,target),dim=1))

    # each `data` is of BATCH_SIZE samples and has shape [batch_size, 4, 128, 128]
    for batch_idx, (data, _,_, feedbacks) in enumerate(train_loader):
        data = Variable(data)
        if train_on_gpu:
            data = data.cuda()

        # push whole batch of data through VAE.forward() to get recon_loss
        x_recon, mu_z, logvar_z, _ = model(data)

        # calculate scalar loss
        loss_recon = criterion_recon(x_recon,data)
        loss_recon *= data.size(1)*data.size(2)*data.size(3)
        loss_recon.div(data.size(0))

        loss_kl = kl_divergence(mu_z,logvar_z)

        #Feedbacks
        deltas = feedbacks[0].cuda().float()
        x_anchors = feedbacks[1].float()
        y_anchors = feedbacks[2].float()
        tensor_anchors = torch.cat((x_anchors,y_anchors),1).cuda() #BatchSize x 2
        loss_feedbacks = weighted_mse_loss(mu_z,tensor_anchors,deltas)

        loss_VAE = loss_recon + model.beta * loss_kl + loss_feedbacks

        optimizer.zero_grad()
        loss_VAE.backward()
        optimizer.step()

        global_VAE_iter.append(loss_VAE.item())
        recon_loss_iter.append(loss_recon.item())
        kl_loss_iter.append(loss_kl.item())

        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss_VAE.item() ),end='\r')

    if (epoch%10==0) or (epoch == 1):
        print('==========> Epoch: {} ==========> Average loss: {:.4f}'.format(epoch, np.mean(global_VAE_iter)))
        print(f'{timer() - start:.2f} seconds elapsed in epoch.')
        print(f'Reconstruction loss : {np.mean(recon_loss_iter):.2f}, KL loss : {np.mean(kl_loss_iter):.2f}')

    return np.mean(global_VAE_iter), np.mean(kl_loss_iter), np.mean(recon_loss_iter)



def train_Simple_VAE(epochs, model, optimizer, train_loader, train_on_gpu=True):
    '''
    Main function to train a VAE model with standard ELBO objective function for a given number of epochs
    An additional term is present is the objective, to force some points to a defined
    anchors with a MSE loss.
    The idendity of anchored points and the anchors are specified in a CSV file
    that is used by a special DataLoader.
    Please refer to human_guidance/feedback_helpers.py and class 'DSpritesDataset' for more info
    Possible to train from scratch or resume a training (simply pass a trained VAE as input)

    The model is not saved automatically, but returned

    Params :
        epochs (int) : Maximum number of epochs
        model (nn.Module) : VAE model to train
        optimizer (optim.Optimizer) : Optimizer used for VAE training
        train_loader (DataLoader) : Dataloader used for training

    Return a pandas DataFrame containing the training history, as well as the trained model
    '''

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()
    history = []

    lr_schedul_VAE = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=25, gamma=0.5)


    for epoch in range(model.epochs+1,model.epochs+epochs+1):
        global_VAE_loss, kl_loss, recon_loss = train_feedback(epoch, model,optimizer, train_loader, train_on_gpu=True)

        history.append([global_VAE_loss, kl_loss, recon_loss])
        model.epochs += 1

        lr_schedul_VAE.step()

    history = pd.DataFrame(
        history,
        columns=['global_VAE_loss', 'kl_loss', 'recon_loss'])


    total_time = timer() - overall_start
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    print('######### TRAINING FINISHED ##########')

    # Attach the optimizer
    model.optimizer = optimizer

    return model, history



# def inference_recon(model, inference_loader, num_img, train_on_gpu):
#     with torch.no_grad():
#         model.eval()
#
#         inference_iter = iter(inference_loader)
#         features, _ = next(inference_iter)
#
#         if train_on_gpu:
#             # make sure this lives on the GPU
#             features = features.cuda()
#         features = Variable(features, requires_grad=False)
#
#         recon, _, _ = model(features)
#         show(make_grid(features[:num_img,:3,:,:],nrow=int(np.sqrt(num_img))),train_on_gpu)
#         show(make_grid(recon[:num_img,:3,:,:],nrow=int(np.sqrt(num_img))),train_on_gpu)
#
#         #To save the img, use save_image from torch utils
