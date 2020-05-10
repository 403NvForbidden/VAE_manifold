# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-06T12:15:33+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-05-07T10:19:00+10:00

'''
File containing main function to train the VAE with proper single cell images dataset
'''
import torch
from torch.autograd import Variable
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from torch import cuda
from torchvision.utils import save_image, make_grid
from networks import VAE
from helpers import plot_latent_space, show

def train(epoch, model, optimizer, train_loader, train_on_gpu):
    # toggle model to train mode
    model.train()
    train_loss = 0
    kl_loss = 0
    recon_loss = 0

    start = timer()

    criterion_recon = nn.BCEWithLogitsLoss(size_average=False).cuda()

    # each `data` is of BATCH_SIZE samples and has shape [batch_size, 4, 128, 128]
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if train_on_gpu:
            data = data.cuda()
        optimizer.zero_grad()

        # push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, logvar, _ = model(data)

        # calculate scalar loss
        recon_lo = criterion_recon(recon_batch,data)
        recon_lo.div(data.size(0))
        kl_lo = model.kl_divergence(mu,logvar)
        loss = recon_lo + model.beta * kl_lo

        loss.backward()
        train_loss += loss.item()
        kl_loss += kl_lo.item()
        recon_loss += recon_lo.item()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data) ),end='\r')
    train_loss /= len(train_loader.dataset)
    kl_loss /= len(train_loader.dataset)
    recon_loss /= len(train_loader.dataset)
    if epoch % 1 == 0:
        print('==========> Epoch: {} ==========> Average loss: {:.4f}'.format(epoch, train_loss))
        print(f'{timer() - start:.2f} seconds elapsed in epoch.')
        print(f'Reconstruction loss : {recon_loss:.4f}, KL loss : {kl_loss:.4f}')


    # visualize reconstrunction, synthesis, and latent space
    if (epoch%1==0) or (epoch == 1):

        fig, ax, fig2, ax2 = plot_latent_space(model,train_loader,train_on_gpu)
        if ax != None and ax2 != None:
            ax.set_title(f'2D Latent Space after {epoch} epochs, 1 color = 1 Well = 1 condition \n Reconstruction loss : {recon_loss:.4f}, KL loss : {kl_loss:.4f}')
            ax2.set_title(f'2D Latent Space after {epoch} epochs,\n Ground truth clusters\' center of mass and 0.7std (51.6%) confidence ellipse are plotted')
        if fig != None :
            fig.show()
        if fig2 != None :
            fig2.show()

        img_grid = make_grid(torch.cat((data[:4,:3,:,:],nn.Sigmoid()(recon_batch[:4,:3,:,:]))), nrow=4, padding=12, pad_value=1)

        plt.figure(figsize=(10,5))
        plt.imshow(img_grid.detach().cpu().permute(1,2,0))
        plt.axis('off')
        plt.title(f'Example data and its reconstruction - epoch {epoch}')
        plt.show()

        samples = torch.randn(8, model.zdim, 1, 1)
        samples = Variable(samples,requires_grad=False).cuda()
        recon = model.decode(samples)
        img_grid = make_grid(nn.Sigmoid()(recon[:,:3,:,:]), nrow=4, padding=12, pad_value=1)

        plt.figure(figsize=(10,5))
        plt.imshow(img_grid.detach().cpu().permute(1,2,0))
        plt.axis('off')
        plt.title(f'Random generated samples - epoch {epoch}')
        plt.show()


    return train_loss, kl_loss, recon_loss

def test(epoch, model, optimizer, test_loader, train_on_gpu, beta):
    # toggle model to test / inference mode

    with torch.no_grad():
        model.eval()
        test_loss = 0
        kl_loss = 0
        recon_loss = 0

        # each data is of BATCH_SIZE (default 128) samples
        for i, (data, _) in enumerate(test_loader):
            if train_on_gpu:
                # make sure this lives on the GPU
                data = data.cuda()

            # we're only going to infer, so no autograd at all required
            data = Variable(data, requires_grad=False)
            recon_batch, mu, logvar = model(data)
            loss, kl, recon = model.loss_function(recon_batch, data, mu, logvar, beta)
            test_loss += loss.item()
            kl_loss += kl.item()
            recon_loss += recon.item()

        test_loss /= len(test_loader.dataset)
        kl_loss /= len(test_loader.dataset)
        recon_loss /= len(test_loader.dataset)
        if epoch % 10 == 0:
            print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss, kl_loss, recon_loss

def beta_set(max_val, start, reach, epoch):
    '''Return the beta weighting factor that reduce the weight of KL loss
    during first epochs.
    Beta takes value 0 to max_val, linearly between epoch 0 to interval
    '''
    if epoch <= start:
        return 0
    if epoch <= reach :
        return (max_val/(reach-start))*(epoch-1) - (start*max_val/(reach-start))
    else :
        return max_val


def train_VAE_model(epochs, model, optimizer, dataloader, train_on_gpu):
    '''
    Params :
        beta_init ([int]) : [max_value, start, reach]
    '''

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    history = []
    overall_start = timer()
    for epoch in range(model.epochs+1,model.epochs+epochs+1):

        # TODO: Set value for beta evolution outside
        #beta = beta_set(max_val=beta_init[0],start=beta_init[1],reach=beta_init[2],epoch=epoch)
        tr_loss, tr_kl, tr_recon = train(epoch, model, optimizer, dataloader['train'], train_on_gpu)
        #te_loss, te_kl, te_recon = test(epoch, model, optimizer, dataloader['val'], train_on_gpu, beta)
        history.append([tr_loss, tr_kl, tr_recon])#, te_loss, te_kl, te_recon])

        model.epochs += 1

    history = pd.DataFrame(
        history,
        columns=['train_loss','train_kl','train_recon'])#,'valid_loss','val_kl','val_recon'])

    total_time = timer() - overall_start
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    print('######### TRAINING FINISHED ##########')

    # Attach the optimizer
    model.optimizer = optimizer

    return model, history


#####################################
### Training of InfoMax
####################################
def train_infoM_epoch(epoch, VAE, MLP, opti_VAE, opti_MLP, train_loader, train_on_gpu):
    '''Train the infoMAX model for one epoch'''
    # toggle model to train mode
    VAE.train()
    global_VAE_loss = 0
    MI_estimation = 0
    MI_estimator_loss = 0
    kl_loss = 0
    recon_loss = 0

    start = timer()

    criterion_recon = nn.BCEWithLogitsLoss(size_average=False).cuda() #more stable than handmade sigmoid as last layer and BCELoss

    # each `data` is of BATCH_SIZE samples and has shape [batch_size, 4, 128, 128]
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if train_on_gpu:
            data = data.cuda()

        #data feed to CNN-VAE
        x_recon, mu_z, logvar_z, z = VAE(data)
        t_xz = MLP(data,z)
        z_perm = VAE.permute_dims(z)
        t_xz_tilda = MLP(data,z_perm)

        #Estimation of the Mutual Info between X and Z
        MI_xz = (t_xz.mean() - (torch.exp(t_xz_tilda -1).mean()))

        loss_recon = criterion_recon(x_recon,data)
        loss_recon.div(data.size(0))
        loss_kl = VAE.kl_divergence(mu_z,logvar_z)

        loss_VAE = loss_recon + VAE.beta * loss_kl - VAE.alpha * MI_xz

        # Step 1 : Optimization of VAE based on the estimation of MI
        opti_VAE.zero_grad()
        loss_VAE.backward(retain_graph=True) #Important argument, we backpropagated two times over MI_xz
        opti_VAE.step()

        MI_loss = -MI_xz

        global_VAE_loss += loss_VAE.item()
        recon_loss += loss_recon.item()
        kl_loss += loss_kl.item()
        MI_estimation += MI_xz.item()
        MI_estimator_loss += MI_loss.item()

        # Step 2 : Optimization of the MLP to improve the MI estimation
        opti_MLP.zero_grad()
        MI_loss.backward()
        opti_MLP.step()


        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss_VAE.item() ),end='\r')
    global_VAE_loss /= len(train_loader)
    recon_loss /= len(train_loader)
    kl_loss /= len(train_loader)
    MI_estimation /= len(train_loader)
    MI_estimator_loss /= len(train_loader)
    if epoch % 1 == 0:
        print('==========> Epoch: {} ==========> Average loss: {:.4f}'.format(epoch, global_VAE_loss))
        print(f'{timer() - start:.2f} seconds elapsed in epoch.')
        print(f'Reconstruction loss : {recon_loss:.2f}, KL loss : {kl_loss:.2f} \n MI : {MI_estimation:.2f} ')


    # visualize reconstrunction, synthesis, and latent space
    if (epoch%1==0) or (epoch == 1):

        fig, ax, fig2, ax2 = plot_latent_space(VAE,train_loader,train_on_gpu)
        if ax != None and ax2 != None:
            ax.set_title(f'2D Latent Space after {epoch} epochs, 1 color = 1 Well = 1 condition \n Reconstruction loss : {recon_loss:.2f}, KL loss : {kl_loss:.2f} \n MI : {MI_estimation:.2f} ')
            ax2.set_title(f'2D Latent Space after {epoch} epochs,\n Ground truth clusters\' center of mass and 0.7std (51.6%) confidence ellipse are plotted')
        if fig != None :
            fig.show()
        if fig2 != None :
            fig2.show()

        img_grid = make_grid(torch.cat((data[:4,:3,:,:],nn.Sigmoid()(x_recon[:4,:3,:,:]))), nrow=4, padding=12, pad_value=1)

        plt.figure(figsize=(10,5))
        plt.imshow(img_grid.detach().cpu().permute(1,2,0))
        plt.axis('off')
        plt.title(f'Example data and its reconstruction - epoch {epoch}')
        plt.show()

        samples = torch.randn(8, VAE.zdim, 1, 1)
        samples = Variable(samples,requires_grad=False).cuda()
        recon = VAE.decode(samples)
        img_grid = make_grid(nn.Sigmoid()(recon[:,:3,:,:]), nrow=4, padding=12, pad_value=1)

        plt.figure(figsize=(10,5))
        plt.imshow(img_grid.detach().cpu().permute(1,2,0))
        plt.axis('off')
        plt.title(f'Random generated samples - epoch {epoch}')
        plt.show()


    return global_VAE_loss, MI_estimation, MI_estimator_loss, kl_loss, recon_loss

def train_InfoMAX_model(epochs,VAE, MLP, opti_VAE, opti_MLP, dataloader, train_on_gpu):
    '''
    Params :
        beta_init ([int]) : [max_value, start, reach]
    '''

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {VAE.epochs} epochs.\n')
    except:
        VAE.epochs = 0
        print(f'Starting Training from Scratch.\n')

    history = []
    overall_start = timer()
    for epoch in range(VAE.epochs+1,VAE.epochs+epochs+1):

        global_VAE_loss, MI_estimation, MI_estimator_loss, kl_loss, recon_loss = train_infoM_epoch(epoch, VAE, MLP, opti_VAE, opti_MLP, dataloader['train'], train_on_gpu)
        history.append([global_VAE_loss, MI_estimation, MI_estimator_loss, kl_loss, recon_loss])

        VAE.epochs += 1

    history = pd.DataFrame(
        history,
        columns=['global_VAE_loss', 'MI_estimation', 'MI_estimator_loss', 'kl_loss', 'recon_loss'])#,'valid_loss','val_kl','val_recon'])

    total_time = timer() - overall_start
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    print('######### TRAINING FINISHED ##########')

    # Attach the optimizer
    VAE.optimizer = opti_VAE
    MLP.optimizer = opti_MLP

    return VAE, MLP, history




def inference_recon(model, inference_loader, num_img, train_on_gpu):
    with torch.no_grad():
        model.eval()

        inference_iter = iter(inference_loader)
        features, _ = next(inference_iter)

        if train_on_gpu:
            # make sure this lives on the GPU
            features = features.cuda()
        features = Variable(features, requires_grad=False)

        recon, _, _ = model(features)
        # TODO: WARNING, we show only 3 first channel to plot, but find a better option !
        show(make_grid(features[:num_img,:3,:,:],nrow=int(np.sqrt(num_img))),train_on_gpu)
        show(make_grid(recon[:num_img,:3,:,:],nrow=int(np.sqrt(num_img))),train_on_gpu)

        #To save the img, use save_image from torch utils
