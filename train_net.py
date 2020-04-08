# @Author: Sacha Haidinger <sachahaidinger>
# @Date:   2020-04-06T12:15:33+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning Methods for Cell Profiling
# @Last modified by:   sachahaidinger
# @Last modified time: 2020-04-07T18:17:08+10:00

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

def train(epoch, model, optimizer, train_loader, train_on_gpu):
    # toggle model to train mode
    model.train()
    train_loss = 0

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
        loss = model.loss_function(recon_batch, data, mu, logvar)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)),end='\r')

    print('==========> Epoch: {} ==========> Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    print(f'{timer() - start:.2f} seconds elapsed in epoch.')
    return train_loss / len(train_loader.dataset)

def test(epoch, model, optimizer, test_loader, train_on_gpu):
    # toggle model to test / inference mode

    with torch.no_grad():
        model.eval()
        test_loss = 0

        # each data is of BATCH_SIZE (default 128) samples
        for i, (data, _) in enumerate(test_loader):
            if train_on_gpu:
                # make sure this lives on the GPU
                data = data.cuda()

            # we're only going to infer, so no autograd at all required
            data = Variable(data, requires_grad=False)
            recon_batch, mu, logvar = model(data)
            test_loss += model.loss_function(recon_batch, data, mu, logvar).item()

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    return test_loss


def train_VAE_model(epochs, model, optimizer, dataloader, train_on_gpu):


    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    history = []
    overall_start = timer()
    for epoch in range(1,epochs+1):

        tr_loss = train(epoch, model, optimizer, dataloader['train'], train_on_gpu)
        te_loss = test(epoch, model, optimizer, dataloader['val'], train_on_gpu)
        history.append([tr_loss,te_loss])

        model.epochs += 1

    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss'])

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
        show(make_grid(features[:num_img,:1,:,:],nrow=int(np.sqrt(num_img))))
        show(make_grid(recon[:num_img,:1,:,:],nrow=int(np.sqrt(num_img))))

        #To save the img, use save_image from torch utils



def show(img):
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
    plt.figure(figsize=(8, 6))
    #for c in ['train_loss', 'valid_loss']:
    #    plt.plot(
    #        history[c], label=c)
    plt.plot(history['train_loss'],color='dodgerblue',label='train_loss')
    plt.plot(history['valid_loss'],color='lightsalmon',label='valid_loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    plt.savefig('outputs/loss_evo_run1.png')



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
