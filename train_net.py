# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-06T12:15:33+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-04-14T16:52:15+10:00

'''
File containing main function to train the VAE with proper single cell images dataset
'''
import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from torch import cuda
from torchvision.utils import save_image, make_grid

def train(epoch, model, optimizer, train_loader, train_on_gpu, beta):
    # toggle model to train mode
    model.train()
    train_loss = 0
    kl_loss = 0
    recon_loss = 0

    start = timer()

    # each `data` is of BATCH_SIZE samples and has shape [batch_size, 4, 128, 128]
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if train_on_gpu:
            data = data.cuda()
        optimizer.zero_grad()

        # push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, logvar = model(data)

        # calculate scalar loss
        loss, kl_lo, recon_lo = model.loss_function(recon_batch, data, mu, logvar, beta)

        loss.backward()
        train_loss += loss.item()
        kl_loss += kl_lo.item()
        recon_loss += recon_lo.item()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)),end='\r')
    train_loss /= len(train_loader.dataset)
    kl_loss /= len(train_loader.dataset)
    recon_loss /= len(train_loader.dataset)
    if epoch % 10 == 0:
        print('==========> Epoch: {} ==========> Average loss: {:.4f}'.format(epoch, train_loss))
        print(f'{timer() - start:.2f} seconds elapsed in epoch.')


    # visualize reconstrunction and synthesis
    if (epoch%15==14):
        img_grid = make_grid(torch.cat((data[:4,:3,:,:],recon_batch[:4,:3,:,:])), nrow=4, padding=12, pad_value=1)

        plt.figure(figsize=(10,5))
        plt.imshow(img_grid.detach().cpu().permute(1,2,0))
        plt.axis('off')
        plt.title(f'Example data and its reconstruction - epoch {epoch}')
        plt.show()

        samples = torch.randn(8, 512, 1, 1)
        samples = Variable(samples,requires_grad=False).cuda()
        recon = model.decode(samples)
        img_grid = make_grid(recon[:,:3,:,:], nrow=4, padding=12, pad_value=1)

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

def beta_set(max_val, interval, epoch):
    '''Return the beta weighting factor that reduce the weight of KL loss
    during first epochs.
    Beta takes value 0 to max_val, linearly between epoch 0 to interval
    '''
    beta = 0
    if epoch <= interval :
        return (max_val/interval)*(epoch-1)
    else :
        return max_val


def train_VAE_model(epochs, model, optimizer, dataloader, train_on_gpu):


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
        beta = beta_set(max_val=1.0,interval=20,epoch=epoch)
        tr_loss, tr_kl, tr_recon = train(epoch, model, optimizer, dataloader['train'], train_on_gpu, beta)
        te_loss, te_kl, te_recon = test(epoch, model, optimizer, dataloader['val'], train_on_gpu, beta)
        history.append([tr_loss, tr_kl, tr_recon, te_loss, te_kl, te_recon, beta])

        model.epochs += 1

    history = pd.DataFrame(
        history,
        columns=['train_loss','train_kl','train_recon','valid_loss','val_kl','val_recon','beta'])

    total_time = timer() - overall_start
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    print('######### TRAINING FINISHED ##########')

    # Attach the optimizer
    model.optimizer = optimizer

    return model, history

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



def show(img, train_on_gpu):
    if train_on_gpu:
        npimg = img.cpu().numpy()
    else:
        npimg = img.numpy()
    plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


def plot_train_result(history):
    """Display training and validation loss evolution over epochs

    Params
    -------
    history : DataFrame
        Panda DataFrame containing train and valid losses

    Return
    --------

    """
    plt.figure(figsize=(20, 20))
    fig, ((ax1 ,ax2),(ax3 ,ax4)) = plt.subplots(2,2)
    #for c in ['train_loss', 'valid_loss']:
    #    plt.plot(
    #        history[c], label=c)
    ax1.plot(history['train_loss'],color='dodgerblue',label='train_loss')
    ax1.plot(history['valid_loss'],color='lightsalmon',label='valid_loss')
    ax1.set_title('General Loss')
    ax2.plot(history['train_kl'],color='dodgerblue',label='train KL loss')
    ax2.plot(history['val_kl'],color='lightsalmon',label='valid KL loss')
    ax2.set_title('KL Loss')
    ax3.plot(history['train_recon'],color='dodgerblue',label='train RECON loss')
    ax3.plot(history['val_recon'],color='lightsalmon',label='valid RECON loss')
    ax3.set_title('Reconstruction Loss')
    ax4.plot(history['beta'],color='dodgerblue',label='Beta')
    ax4.set_title('KL Cost Weight')

    plt.legend()


def save_checkpoint(model, path):
    """Save a vgg16 model to path.
        Useful to reuse later on a model already trained on our specific data

    Params
    --------
        model (PyTorch model): model to save
        path (str): location to save model. must end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    # Basic details, as mapping id to classe and number of epochs already trained
    checkpoint = {
        'epochs': model.epochs,
    }

    checkpoint['state_dict'] = model.state_dict() #Weights

    # Add the optimizer
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    torch.save(checkpoint, path)


def load_checkpoint(model, path):
    """Load a VAE network, pre-trained on single cell images

    Params
    --------
        path (str): saved model checkpoint. Must end in '.pth'

    Returns
    --------
        None, load the `model` from `path`

    """
    # Load in checkpoint
    train_on_gpu = cuda.is_available()
    if train_on_gpu:
        checkpoint = torch.load(path)
    else :
        checkpoint = torch.load(path,map_location='cpu')  #TODO Attention, bug si reload depuis autre part que GPU !!!
        ## TODO: inspect why


    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')

    if train_on_gpu:
        model = model.to('cuda')

    model.epochs = checkpoint['epochs']

    # Optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer
