# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-04-20T08:16:22+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-05-07T10:28:57+10:00

'''File containing function to visualize data or to save it'''
import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch import cuda
from torchvision.utils import save_image, make_grid
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


############################
######## VISUALIZATION
############################

def plot_latent_space(model, dataloader, train_on_gpu):

    # colors = ["windows blue", "amber",
    #   "greyish", "faded green",
    #   "dusty purple","royal blue","lilac",
    #   "salmon","bright turquoise",
    #   "dark maroon","light tan",
    #   "orange","orchid",
    #   "sandy","topaz",
    #   "fuchsia","yellow",
    #   "crimson","cream","goldenrod",
    #   "lime", "lightseagreen","violet","crimson",
    #   "navy","blanchedalmond","darkorange"
    #   ]

    labels_list = []
    z_list = []

    if model.zdim > 3 :
        print(f'Latent space is >3D ({model.zdim} dimensional), no visualization is provided')
        return None, None, None, None

    # TODO: Use a dataloader which do not 'drop last' for inference
    for i, (data, labels) in enumerate(dataloader):
        if train_on_gpu:
            # make sure this lives on the GPU
            data = data.cuda()
        with torch.no_grad():
            model.eval()

            data = Variable(data, requires_grad=False)

            #The mean can be taken as the most likely z
            z, _ = model.encode(data)
            z = z.view(-1,model.zdim)
            z_list.append((z.data).cpu().numpy())
            labels_list.append(labels.numpy())

        print(f'In progress...{i*len(data)}/{len(dataloader.dataset)}',end='\r')

    z_points = np.concatenate(z_list,axis=0) # datasize x 3
    true_label = np.concatenate(labels_list,axis=0)

    if model.zdim == 3:
        fig = plt.figure()
        ax = Axes3D(fig)

        for i in np.unique(true_label):
            ax.scatter(z_points[true_label==i,0],z_points[true_label==i,1],z_points[true_label==i,2],s=5)

        return fig, fig

    if model.zdim == 2:
        fig = plt.figure(figsize=(12,12))
        ax = plt.subplot(111)

        #if 2D plot a separated plot with confidence ellipse
        fig2, ax_ellipse = plt.subplots(figsize=(12,12))

        cmap1 = plt.get_cmap('tab20')
        colors1 = cmap1(np.linspace(0,1.0,20))
        cmap2 = plt.get_cmap('Set3')
        colors2 = cmap2(np.linspace(0,1.0,10))
        colors = np.concatenate((colors1,colors2),0)

        for i in np.unique(true_label):
            ax.scatter(z_points[true_label==i,0],z_points[true_label==i,1],s=5,color=colors[i])

            mu_1 = np.mean(z_points[true_label==i,0])
            mu_2 = np.mean(z_points[true_label==i,1])

            confidence_ellipse(z_points[true_label==i,0], z_points[true_label==i,1], ax_ellipse,n_std=0.7,
                alpha=0.5,facecolor=colors[i], edgecolor=colors[i] , zorder=0)
            ax_ellipse.scatter(mu_1, mu_2,marker='X', s=50, color=colors[i])

        return fig, ax, fig2, ax_ellipse


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def show(img, train_on_gpu):
    if train_on_gpu:
        npimg = img.cpu().numpy()
    else:
        npimg = img.numpy()
    plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


def plot_train_result(history, infoMAX = False, only_train_data=True):
    """Display training and validation loss evolution over epochs

    Params
    -------
    history : DataFrame
        Panda DataFrame containing train and valid losses

    Return
    --------

    """
    fig, ((ax1 ,ax2),(ax3 ,ax4)) = plt.subplots(2,2,figsize=(10, 10))
    #for c in ['train_loss', 'valid_loss']:
    #    plt.plot(
    #        history[c], label=c)

    if only_train_data:

        ax1.plot(history['train_loss'],color='dodgerblue',label='Global loss')
        ax1.set_title('General Loss')
        ax2.plot(history['train_kl'],color='dodgerblue',label='KL loss')
        ax2.set_title('KL Loss')
        ax3.plot(history['train_recon'],color='dodgerblue',label='RECON loss')
        ax3.set_title('Reconstruction Loss')
        #ax4.plot(history['beta'],color='dodgerblue',label='Beta')
        #ax4.set_title('KL Cost Weight')

    if only_train_data and infoMAX:
        ax1.plot(history['global_VAE_loss'],color='dodgerblue',label='Global Loss')
        ax1.set_title('global_VAE_loss')
        ax2.plot(history['MI_estimator_loss'][1:],color='dodgerblue',label='MLP Loss')
        ax2.plot(history['MI_estimation'][1:],color='lightsalmon',label='MI Estimation')
        ax2.set_title('Mutual Information')
        ax3.plot(history['recon_loss'],color='dodgerblue',label='Reconstruction Loss')
        ax3.set_title('recon_loss')
        ax4.plot(history['kl_loss'],color='dodgerblue',label='Fit to prior loss')
        ax4.set_title('kl_loss')

    # else:
    #
    #     ax1.plot(history['train_loss'],color='dodgerblue',label='train_loss')
    #     ax1.plot(history['valid_loss'],color='lightsalmon',label='valid_loss')
    #     ax1.set_title('General Loss')
    #     ax2.plot(history['train_kl'],color='dodgerblue',label='train KL loss')
    #     ax2.plot(history['val_kl'],color='lightsalmon',label='valid KL loss')
    #     ax2.set_title('KL Loss')
    #     ax3.plot(history['train_recon'],color='dodgerblue',label='train RECON loss')
    #     ax3.plot(history['val_recon'],color='lightsalmon',label='valid RECON loss')
    #     ax3.set_title('Reconstruction Loss')
    #     ax4.plot(history['beta'],color='dodgerblue',label='Beta')
    #     ax4.set_title('KL Cost Weight')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    return fig






############################
######## SAVING
############################

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
        'zdim' : model.zdim,
        'channels' : model.channels,
        'layer_count' : model.layer_count,
        'base' : model.base,
        'loss' : model.loss,
        'input_size' : model.input_size,
        'model_type' : model.model_type
    }

    checkpoint['state_dict'] = model.state_dict() #Weights

    # Add the optimizer
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    torch.save(checkpoint, path)

def save_brute(model, path):
    '''Save the entire model
    For fast development purpose only'''

    torch.save(model,path)


def load_brute(path):
    '''To reload entire model
    For fast development purpose only'''

    return torch.load(path)


def load_checkpoint(mpath):
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


    if checkpoint['model_type'] == 'VAE_CNN_vanilla' :
        model = VAE(zdim=checkpoint['zdim'],channels=checkpoint['channels'],base=checkpoint['base'],loss=checkpoint['loss'],layer_count=checkpoint['layer_count'],input_size=checkpoint['input_size'])


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
