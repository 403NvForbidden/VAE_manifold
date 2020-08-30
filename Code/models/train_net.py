# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-06T12:15:33+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-29T12:36:15+10:00

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
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import torch
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
from torch import cuda
from torchvision.utils import save_image, make_grid

from networks import VAE
from helpers import plot_latent_space, show, EarlyStopping
from infoMAX_VAE import infoNCE_bound


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

        loss_kl = model.kl_divergence(mu_z,logvar_z)
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
            loss_kl = model.kl_divergence(mu_z,logvar_z)

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
        global_VAE_loss, kl_loss, recon_loss = train(epoch, model,optimizer, train_loader, each_iteration, train_on_gpu)
        global_VAE_loss_val, kl_loss_val, recon_loss_val = test(epoch, model,optimizer,valid_loader, each_iteration, train_on_gpu)

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
        scores = MLP(data,z)

        #Estimation of the Mutual Info between X and Z
        MI_xz = infoNCE_bound(scores)

        loss_recon = criterion_recon(x_recon,data)
        loss_recon *= data.size(1)*data.size(2)*data.size(3)
        loss_recon.div(data.size(0))
        loss_kl = VAE.kl_divergence(mu_z,logvar_z)

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
            loss_kl = VAE.kl_divergence(mu_z,logvar_z)

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
    early_stopping = EarlyStopping(patience=30,verbose=True,path=saving_path)
    lr_schedul_VAE = torch.optim.lr_scheduler.StepLR(optimizer=opti_VAE, step_size=40, gamma=0.6)
    lr_schedul_MLP = torch.optim.lr_scheduler.StepLR(optimizer=opti_MLP, step_size=40, gamma=0.6)

    for epoch in range(VAE.epochs+1,VAE.epochs+epochs+1):
        global_VAE_loss, MI_estimation, MI_estimator_loss, kl_loss, recon_loss = train_infoM_epoch(epoch, VAE, MLP, opti_VAE, opti_MLP, train_loader, each_iteration, train_on_gpu)
        global_VAE_loss_val, MI_estimation_val, MI_estimator_loss_val, kl_loss_val, recon_loss_val = test_infoM_epoch(epoch, VAE, MLP, opti_VAE, opti_MLP, valid_loader, each_iteration, train_on_gpu)

        #ealy stopping takes the validation loss to check if it has decereased,
        #if so, model is saved, if not for 'patience' time in a row, the training loop is broken
        early_stopping(global_VAE_loss_val,VAE,MLP)
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

        loss_kl = model.kl_divergence(mu_z,logvar_z)

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
