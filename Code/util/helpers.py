# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-04-20T08:16:22+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-26T16:58:12+10:00

"""File containing various functions to visualize latent representations learnt
from a model and to save / load model. """

###############################
####### Imports ###############
###############################
import sys, os
import warnings

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.gridspec import GridSpec
import matplotlib.colors
import matplotlib.cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch import cuda
from torchvision.utils import save_image, make_grid


def make_path(path, msg=None):
    if not path == '' and not os.path.isdir(path):
        try:
            os.mkdir(path)
            if msg: print(msg)
            print(f'created path:=======>{path}')
        except FileExistsError as e:
            warnings.warn(f'CANNOT create path {path}, because {e}')

##############################################
######## Match Latent Code and Ground Truth
##############################################
def meta_MNIST(model, dataloader, device='cpu'):
    z_list, labels_list = [], []
    for i, (data, labels) in enumerate(dataloader):
        # Extract unique cell id from file_names
        data = data.to(device)
        model.eval()
        with torch.no_grad():
            # if with_rawdata:
            #     raw_data = data.view(data.size(0),-1) #B x HxWxC
            #     list_of_tensors.append(raw_data.data.cpu().numpy())

            data = Variable(data, requires_grad=False)
            z, _ = model.encode(data)
            z = z.view(-1, model.zdim)
            z_list.append((z.data).cpu().numpy())
            labels_list.append(labels.numpy())

        print(f'In progress...{i * len(data)}/{len(dataloader.dataset)}', end='\r')

    z_points = np.concatenate(z_list, axis=0)  # datasize
    true_label = np.concatenate(labels_list, axis=0)
    df = pd.DataFrame(z_points, columns=['x', 'y', 'z'])
    df["GT"] = pd.Series(true_label)

    ### plot ###
    ##### Fig 1 : Plot each single cell in latent space with GT cluster labels
    traces = []
    for i in np.unique(true_label):  # range(3,5): # TODO: change it back to original form
        df_filter = df[df['GT'] == i]
        scatter = go.Scatter3d(x=df_filter['x'].values,
                               y=df_filter['y'].values,
                               z=df_filter['z'].values,
                               mode='markers',
                               marker=dict(size=3, opacity=1),
                               name=f'Cluster {i}')
        traces.append(scatter)
    layout = dict(title='Latent Representation, colored by GT clustered')
    fig_3d_1 = go.Figure(data=traces, layout=layout)
    fig_3d_1.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=True, legend=dict(y=-.1))

    # TODO
    return fig_3d_1


def get_raw_data(dataloader, MetaData_csv):
    list_of_tensors, id_list = [], []

    for i, (data, labels, file_names) in enumerate(dataloader):
        # Extract unique cell id from file_names
        id_list.append([file_name for file_name in file_names])

        raw_data = data.view(data.size(0), -1)  # B x HxWxC
        list_of_tensors.append(raw_data)

    cols = ['feature' + str(i) for i in range(raw_data.shape[1])]
    raw_data = np.concatenate(list_of_tensors, axis=0)
    rawdata_frame = pd.DataFrame(data=raw_data[0:, 0:],
                                 index=[i for i in range(raw_data.shape[0])],
                                 columns=cols)
    rawdata_frame['Unique_ID'] = np.nan
    rawdata_frame.Unique_ID = list(itertools.chain.from_iterable(id_list))
    rawdata_frame = rawdata_frame.sort_values(by=['Unique_ID'])

    MetaData_csv['Unique_ID'] = MetaData_csv['Unique_ID'].astype(str)
    rawdata_frame['Unique_ID'] = rawdata_frame['Unique_ID'].astype(str)
    assert np.all(np.unique(rawdata_frame.Unique_ID.values) == np.unique(MetaData_csv.Unique_ID.values)), "Inference dataset doesn't match with csv metadata"
    # align with metadata
    MetaData_csv = MetaData_csv.join(rawdata_frame.set_index('Unique_ID'), on='Unique_ID')
    imgs = MetaData_csv[cols].values
    return torch.from_numpy(imgs.reshape(imgs.shape[0], data.shape[1], data.shape[2], data.shape[3])).float()


def metadata_latent_space(model, dataloader, device, GT_csv_path, save_csv=False, with_rawdata=False,
                          csv_path='no_name_specified.csv'):
    '''
    Once a VAE model is trained, take its predictions (3D latent code) and store it in a csv
    file alongside the ground truth information. Useful for plots and downstream analyses.

    Params :
        - model (nn.Module) :  trained Pytorch VAE model that will produce latent codes of dataset
        - inder_dataloader (DataLoader) : Dataloader that iterates the dataset by batch
        - train_on_gpu (boolean) : Wheter infer latent codes on GPU or not.
        - GT_csv_path (string) : path to a csv file that contains all the ground truth information
                to keep alongside the latent codes. A column 'Unique_ID' (the name of the tiff files)
                needs to be present to match latent code and ground truth
        - save_csv (boolean) : If False, the new Panda DataFrame in only returned.
        - csv_path (string) : the path where to store a new csv file that contains the matched
                latent code and ground truth. Ignored if 'save_csv' is False
        - with_rawdata (boolean): If True, the raw data (images pixels) is also save in the csv file

    Return :
        - MetaData_csv (Pandas DataFrame) : DataFrame that contains both latent codes and ground truth, matched
    '''
    labels_list = []
    z_list = []
    id_list = []
    list_of_tensors = []  # Store raw_data for performance metrics

    model.to(device)
    if model.zdim > 3: warnings.warn(f'Latent space is >3D ({model.zdim} dimensional), no visualization is provided')

    ###### Iterate throughout inference dataset #####
    #################################################
    for i, (data, labels, file_names) in enumerate(dataloader):
        # Extract unique cell id from file_names
        id_list.append([file_name for file_name in file_names])
        data = data.to(device)
        model.eval()
        with torch.no_grad():
            if with_rawdata:
                raw_data = data.view(data.size(0), -1)  # B x HxWxC
                list_of_tensors.append(raw_data.data.cpu().numpy())

            data = Variable(data, requires_grad=False)
            z, _, _ = model.inference(model.encode(data))
            z = z.view(-1, model.zdim)
            z_list.append((z.data).cpu().numpy())
            labels_list.append(labels.numpy())

        print(f'In progress...{i * len(data)}/{len(dataloader.dataset)}', end='\r')

    ###### Matching samples to metadata info #####
    #################################################
    unique_ids = list(itertools.chain.from_iterable(id_list))

    ###### Store raw data in a separate data frame #####
    if with_rawdata:
        raw_data = np.concatenate(list_of_tensors, axis=0)
        rawdata_frame = pd.DataFrame(data=raw_data[0:, 0:],
                                     index=[i for i in range(raw_data.shape[0])],
                                     columns=['feature' + str(i) for i in range(raw_data.shape[1])])
        rawdata_frame['Unique_ID'] = np.nan
        rawdata_frame.Unique_ID = unique_ids
        rawdata_frame = rawdata_frame.sort_values(by=['Unique_ID'])

    ##### Store latent code in a temporary dataframe #####
    z_points = np.concatenate(z_list, axis=0)  # datasize x 3
    # true_label = np.concatenate(labels_list,axis=0)

    temp_matching_df = pd.DataFrame(z_points, columns=[f'z{n}' for n in range(model.zdim)])
    temp_matching_df['Unique_ID'] = unique_ids
    temp_matching_df['Unique_ID'] = temp_matching_df['Unique_ID'].astype(str)
    temp_matching_df = temp_matching_df.sort_values(by=['Unique_ID'])

    ##### Load Ground Truth information about the dataset #####
    MetaData_csv = pd.read_csv(GT_csv_path).sort_values(by=['Unique_ID'])
    MetaData_csv['Unique_ID'] = MetaData_csv['Unique_ID'].astype(str)
    ##### Match latent code information with ground truth info #####
    # assert np.all(temp_matching_df.Unique_ID.values == MetaData_csv.Unique_ID.values), "Inference dataset doesn't match with csv metadata"
    MetaData_csv = pd.merge(MetaData_csv, temp_matching_df.set_index('Unique_ID'), how='outer', on=["Unique_ID"])

    ##### Match raw data information with ground truth info #####
    if with_rawdata:
        assert np.all(
            rawdata_frame.Unique_ID.values == MetaData_csv.Unique_ID.values), "Inference dataset doesn't match with csv metadata"
        MetaData_csv = MetaData_csv.join(rawdata_frame.set_index('Unique_ID'), on='Unique_ID')

    #### Save Final CSV file #####
    if save_csv:
        MetaData_csv.to_csv(csv_path, index=False)
        print(f'Final CSV saved to : {csv_path}')

    return MetaData_csv.dropna().reindex()


##############################################
######## Visualization
##############################################
def single_reconstruciton(loader, model, save_path, device, num_img=12, gen=True, logger=None):
    data, _, _ = next(iter(loader))
    data = Variable(data[:num_img], requires_grad=False).to(device)
    model.to(device)

    ### reconstruction
    x_recon_1, _, _, z_1 = model(data)

    img_grid = make_grid(
        torch.cat(
            (data[:, :3, :, :], torch.sigmoid(x_recon_1[:, :3, :, :]))),
        nrow=num_img, padding=12,
        pad_value=1)

    if logger:
        logger.experiment.add_image("Image reconstruction", img_grid)
    plt.figure(figsize=(10, 5))
    plt.imshow(img_grid.detach().cpu().permute(1, 2, 0))
    plt.axis('off')
    plt.title(f'Reconstruction Samples')
    plt.savefig(os.path.join(save_path, 'reconSamples.png'))

    if gen:
        ### TODO: larger grid
        samples_1 = Variable(torch.randn(num_img, model.zdim, 1, 1), requires_grad=False).to(device)
        recon_1 = model.decode(samples_1)
        img_grid = make_grid(torch.sigmoid(recon_1[:, :3, :, :]),
                             nrow=num_img, padding=12, pad_value=1)

        if logger:
            logger.experiment.add_image("Image generation", img_grid)

    plt.figure(figsize=(10, 5))
    plt.imshow(img_grid.detach().cpu().permute(1, 2, 0))
    plt.axis('off')
    plt.title(f'Random generated samples')
    plt.savefig(os.path.join(save_path, 'generatedSamples.png'))


def double_reconstruciton(loader, model, save_path, device, num_img=12, gen=True, logger=None):
    print("Plotting")

    data, _, _ = next(iter(loader))
    data = Variable(data[:num_img], requires_grad=False).to(device)
    model.to(device)

    ### reconstruction
    x_recon_hi, _, _, _, x_recon_lo, _, _, _ = model(data)

    img_grid_recon = make_grid(
        torch.cat((data[:, :3, :, :], torch.sigmoid(x_recon_hi[:, :3, :, :]), torch.sigmoid(x_recon_lo[:, :3, :, :]))),
        nrow=num_img, padding=12,
        pad_value=1
    )
    if logger:
        logger.experiment.add_image("Image reconstruction", img_grid_recon)
    plt.figure(figsize=(10, 5))
    plt.imshow(img_grid_recon.detach().cpu().permute(1, 2, 0))
    plt.axis('off')
    plt.title(f'Reconstruction samples')
    plt.savefig(os.path.join(save_path, 'reconstruction.png'))

    if gen:
        ### TODO: larger grid
        samples_aux = Variable(torch.randn(num_img, model.zdim_aux, 1, 1), requires_grad=False).to(device)
        samples = Variable(torch.randn(num_img, model.zdim, 1, 1), requires_grad=False).to(device)

        gen_hi = model.decode_aux(samples_aux)
        gen_lo = model.decode(samples)
        img_grid_gen = make_grid(
            torch.cat((torch.sigmoid(gen_hi[:, :3, :, :]), torch.sigmoid(gen_lo[:, :3, :, :]))),
            nrow=num_img,
            padding=12,
            pad_value=1
        )

        if logger:
            logger.experiment.add_image("High fidelity generation (top), and Low fidelity generation (bot)",
                                        img_grid_gen)
        plt.figure(figsize=(10, 5))
        plt.imshow(img_grid_gen.detach().cpu().permute(1, 2, 0))
        plt.axis('off')
        plt.title(f'Random generated samples')
        plt.savefig(os.path.join(save_path, 'generatedSamples.png'))


def plot_from_csv(path_to_csv, low_dim_names=['VAE_x_coord', 'VAE_y_coord', 'VAE_z_coord'], GT_col='GT_label', dim=3,
                  column=None, as_str=False):
    """
    Plot on a plotly figure the latent space produced by any methods, already stored in a csv file.

    Params :
        - path_to_csv (string or pandas DataFrame) : path to a csv file or pandas DataFrame that contains the latent codes
                produced by a considered methods.
        - low_dim_names ([string]) : List of columns names that store the latent code to plot
        - dim (int) : Dimensionality of the latent code
        - num_class (int) : Number of different class. If a number is given, latent code will be color-coded based on the class
                identity (from 1 to num_class) given in the column 'GT_Label'
        - column (string) : Column name that store a ground truth that can be used to color-coded the latent space, rather
                than using the 'GT_Label' class identity
        - as_str (boolean) : If a 'column' name is given, and as_str is set to True, every unique value is considered as a
                class and will be colored by a different color. If set to False and values are int or float, a colorbar is used.

    Example :
    plot_from_csv(path_to_csv=...,low_dim_names=...,num_class=7) : Latent Codes color-coded based on int values (1-7) stored in 'GT_Label' column
    plot_from_csv(path_to_csv=...,low_dim_names=...,column='Shape_Factor',as_str=False) : Latent Codes color-coded based on a color bar that cover the range of ground truth 'Shape_Factor'
    """
    if isinstance(path_to_csv, str):
        MetaData_csv = pd.read_csv(path_to_csv)
    else:
        MetaData_csv = path_to_csv

    if dim == 3:
        if column == None:
            ##### Fig 1 : Plot each single cell in latent space with GT cluster labels
            traces = []
            for i in np.unique(
                    MetaData_csv[GT_col]):  # [2, 3, 4]:  # range(3,5): # TODO: change it back to original form
                selected_rows = MetaData_csv[MetaData_csv[GT_col] == i]
                scatter = go.Scatter3d(x=selected_rows[low_dim_names[0]].values,
                                       y=selected_rows[low_dim_names[1]].values,
                                       z=selected_rows[low_dim_names[2]].values,
                                       mode='markers',
                                       marker=dict(size=3, opacity=1),
                                       name=f'Cluster {i}')
                traces.append(scatter)
            layout = dict(title='Latent Representation, colored by GT clustered')
            fig_3d_1 = go.Figure(data=traces, layout=layout)
            fig_3d_1.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=True, legend=dict(y=-.1))

            return fig_3d_1
        else:
            # Color from the info store in 'column' pass in argument
            if as_str:
                MetaData_csv[column] = MetaData_csv[column].astype(str)
            fig = px.scatter_3d(MetaData_csv, x=low_dim_names[0], y=low_dim_names[1], z=low_dim_names[2], color=column,
                                color_discrete_sequence=px.colors.qualitative.T10 + px.colors.qualitative.Alphabet)
            fig.update_traces(marker=dict(size=3))
            return fig

    elif dim == 2:

        traces = []
        MS = MetaData_csv
        for i in np.unique(
                    MetaData_csv['GT_dataset']):
            scatter = go.Scatter(x=MS[MetaData_csv[GT_col] == i][low_dim_names[0]].values,
                                 y=MS[MetaData_csv[GT_col] == i][low_dim_names[1]].values,
                                 mode='markers',
                                 marker=dict(size=3, opacity=0.8),
                                 name=f'Cluster {i}')
            traces.append(scatter)
        layout = dict(title='Latent Representation, colored by GT clustered')
        fig_2d_1 = go.Figure(data=traces, layout=layout)
        fig_2d_1.update_layout(margin=dict(l=0, r=0, b=0, t=0), showlegend=True)

        return fig_2d_1


def plot_train_result(history, best_epoch=None, save_path=None):
    """Display training and validation loss evolution over epochs

    Params
    -------
    history : DataFrame
        Panda DataFrame containing train and valid losses
    best_epoch : int
        If early stopping is used, display the last saved model
    save_path : string
        Path to save the figure
    infoMax : boolean
        If True, an additional loss composant (Mutual Information) is plotted

    Return a matplotlib Figure
    --------
    """
    print("===========>>>>>>>>> PLOTING")
    # columns=['VAE_loss', 'kl_1', 'kl_2', 'recon_1', 'recon_2', 'VAE_loss_val', 'kl_val_1', 'kl_val_2', 'recon_val_1', 'recon_val_2']
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(2, 1, 1)  # Frist full row
    ax2 = fig.add_subplot(2, 2, 3)  # bottom left on 4x4 grid
    ax3 = fig.add_subplot(2, 2, 4)  # bottom right on a 4x4 grid
    #  plot the overall loss
    ax1.plot(history['VAE_loss'], color='dodgerblue', label='train')
    ax1.plot(history['VAE_loss_val'], linestyle='--', color='dodgerblue', label='test')
    ax1.set_title('Global VAE Loss')

    if best_epoch != None:
        ax1.axvline(best_epoch, linestyle='--', color='r', label='Early stopping')

    ax2.set_title('Reconstruction Loss')
    ax2.plot(history['recon_1'], color='dodgerblue', label='VAE_1')
    ax2.plot(history['recon_2'], color='lightsalmon', label='VAE_2')
    ax2.plot(history['recon_val_1'], linestyle='--', color='dodgerblue', label='VAE_1_val')
    ax2.plot(history['recon_val_2'], linestyle='--', color='lightsalmon', label='VAE_2_val')

    ax3.set_title('Fit to Prior')
    ax3.plot(history['kl_1'], color='dodgerblue', label='train')
    ax3.plot(history['kl_2'], color='lightsalmon', label='train')
    ax3.plot(history['kl_val_1'], linestyle='--', color='dodgerblue', label='test')
    ax3.plot(history['kl_val_2'], linestyle='--', color='lightsalmon', label='test')

    ax1.legend()
    ax2.legend()
    ax3.legend()

    if save_path != None:
        plt.savefig(save_path + 'los_evolution.png')

    return fig


############################
######## SAVING & STOPPING
############################

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    This class is obtained from Bjarten Implementation (GitHub open source), thanks to him"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the models to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.stop_epoch = 0
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model_epoch):

        score = -val_loss

        if self.best_score is None or score > self.best_score + self.delta:  # loss decreased
            self.best_score = score
            # update the min loss so far
            self.val_loss_min = val_loss
            self.counter = 0
            # update the best epoch
            self.stop_epoch = model_epoch
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        else:  # loss not decreasing
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                # end training
                self.early_stop = True

    def save_model(self, models):
        # input: tuple
        if self.path == '':  # dont save
            print("====>NOT saving")
            return

        if len(models) == 2:
            print(f'========Saving 2 stsage VAE model========')
            VAE1, VAE2 = models
            # Save both modelsmetadata_latent_space
            save_brute(VAE1, self.path + 'VAE_1.pth')
            save_brute(VAE2, self.path + 'VAE_2.pth')
        elif len(models) == 4:
            print(f'========Saving 2 stsage infoMAxVAE model========')
            VAE1, MLP1, VAE2, MLP2 = models
            # Save both models
            save_brute(VAE1, self.path + 'VAE_1.pth')
            save_brute(VAE2, self.path + 'VAE_2.pth')
            save_brute(MLP1, self.path + 'MLP_1.pth')
            save_brute(MLP2, self.path + 'MLP_2.pth')


def save_checkpoint(model, path):
    """Save a NN model to path.

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
        'zdim': model.zdim,
        'channels': model.channels,
        'layer_count': model.layer_count,
        'base': model.base,
        'loss': model.loss,
        'input_size': model.input_size,
        'model_type': model.model_type
    }
    checkpoint['state_dict'] = model.state_dict()  # Weights

    # Add the optimizer
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    torch.save(checkpoint, path)


def save_brute(model, path):
    """Save the entire model
    For fast development purpose only"""

    torch.save(model, path)


def load_brute(path):
    """To reload entire model
    For fast development purpose only"""

    return torch.load(path)


def load_checkpoint(path):
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
    else:
        checkpoint = torch.load(path, map_location='cpu')  # If saved from GPU, need to be reload on GPU as well !

    if checkpoint['model_type'] == 'VAE_CNN_vanilla':
        model = betaVAE(zdim=checkpoint['zdim'], channels=checkpoint['channels'], base=checkpoint['base'],
                        loss=checkpoint['loss'], layer_count=checkpoint['layer_count'],
                        input_size=checkpoint['input_size'])

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


#######################################
######## Old visualization Function
#######################################

def plot_latent_space(model, dataloader, train_on_gpu):
    """
    Simple plot of the latent representation. To plot every XXX epochs during training for instance
    """

    labels_list = []
    z_list = []

    if model.zdim > 3:
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

            # The mean can be taken as the most likely z
            z, _ = model.encode(data)
            z = z.view(-1, model.zdim)
            z_list.append((z.data).cpu().numpy())
            labels_list.append(labels.numpy())

        print(f'In progress...{i * len(data)}/{len(dataloader.dataset)}', end='\r')

    z_points = np.concatenate(z_list, axis=0)  # datasize x 3
    true_label = np.concatenate(labels_list, axis=0)

    if model.zdim == 3:
        fig = plt.figure(figsize=(10, 10))
        # ax = Axes3D(fig)
        ax = plt.axes(projection='3d')

        cmap1 = plt.get_cmap('tab20')
        colors1 = cmap1(np.linspace(0, 1.0, 20))
        cmap2 = plt.get_cmap('Set3')
        colors2 = cmap2(np.linspace(0, 1.0, 10))
        colors = np.concatenate((colors1, colors2), 0)
        zorder = 100
        for i in np.unique(true_label):
            zorder -= 10
            if i + 1 == 7:  # Do not plot process 7 for more readability
                continue
            ax.scatter3D(z_points[true_label == i, 0], z_points[true_label == i, 1], z_points[true_label == i, 2], s=5,
                         color=colors[i], zorder=zorder, label=f'Process_{i + 1}')
        ax.legend()
        return fig, ax, fig, ax

    if model.zdim == 2:
        fig = plt.figure(figsize=(12, 12))
        ax = plt.subplot(111)

        # if 2D plot a separated plot with confidence ellipse
        fig2, ax_ellipse = plt.subplots(figsize=(12, 12))

        cmap1 = plt.get_cmap('tab20')
        colors1 = cmap1(np.linspace(0, 1.0, 20))
        cmap2 = plt.get_cmap('Set3')
        colors2 = cmap2(np.linspace(0, 1.0, 10))
        colors = np.concatenate((colors1, colors2), 0)

        zorder = 100
        for i in np.unique(true_label):
            zorder -= 10
            ax.scatter(z_points[true_label == i, 0], z_points[true_label == i, 1], s=5, color=colors[i], zorder=zorder,
                       label=f'Process_{i + 1}')

            mu_1 = np.mean(z_points[true_label == i, 0])
            mu_2 = np.mean(z_points[true_label == i, 1])

            confidence_ellipse(z_points[true_label == i, 0], z_points[true_label == i, 1], ax_ellipse, n_std=0.7,
                               alpha=0.5, facecolor=colors[i], edgecolor=colors[i], zorder=0)
            ax_ellipse.scatter(mu_1, mu_2, marker='X', s=50, color=colors[i], zorder=zorder, label=f'Process_{i + 1}')

        ax.legend()
        ax_ellipse.legend()
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
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
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
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


# imports
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn


def get_new_fig(fn, figsize=[9, 9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()  # Get Current Axis
    ax1.cla()  # clear existing plot
    return fig1, ax1


#

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = [];
    text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:, col]
    ccl = len(curr_column)

    # last line  and/or last column
    if (col == (ccl - 1)) or (lin == (ccl - 1)):
        # tots and percents
        if (cell_val != 0):
            if (col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif (col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif (lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%' % (per_ok), '100%'][per_ok == 100]

        # text to DEL
        text_del.append(oText)

        # text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d' % (cell_val), per_ok_s, '%.2f%%' % (per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy();
        dic['color'] = 'g';
        lis_kwa.append(dic);
        dic = text_kwargs.copy();
        dic['color'] = 'r';
        lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y - 0.3), (oText._x, oText._y), (oText._x, oText._y + 0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            # print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        # print '\n'

        # set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if (col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if (per > 0):
            txt = '%s\n%.2f%%' % (cell_val, per)
        else:
            if (show_null_values == 0):
                txt = ''
            elif (show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        # main diagonal
        if (col == lin):
            # set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del


#

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append(df_cm[c].sum())
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append(item_line[1].sum())
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col
    # print ('\ndf_cm:\n', df_cm, '\n\b\n')


#

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=11,
                                 lw=0.5, cbar=False, figsize=[8, 8], show_null_values=0, pred_val_axis='y'):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if (pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    # this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    # thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    # set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    # iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist())
    text_add = [];
    text_del = [];
    posi = -1  # from left to right, bottom to top.
    for t in ax.collections[0].axes.texts:  # ax.texts:
        pos = np.array(t.get_position()) - [0.5, 0.5]
        lin = int(pos[1]);
        col = int(pos[0]);
        posi += 1
        # print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        # set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    # remove the old ones
    for item in text_del:
        item.remove()
    # append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    # titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  # set layout slim
    # plt.show()

    return fig


#

def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="GnBu",
                                    fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[8, 8], show_null_values=0,
                                    pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    # data
    # if columns == None:
    #     # labels axis integer:
    #     ##columns = range(1, len(np.unique(y_test))+1)
    #     # labels axis string:
    #     from string import ascii_uppercase
    #     columns = ['class %s' % (i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]

    confm = confusion_matrix(y_test, predictions)
    # cmap = 'GnBu';
    # fz = 11;
    # figsize=[9,9];
    # show_null_values = 2
    df_cm = DataFrame(confm, index=columns, columns=columns)
    return pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values,
                                        pred_val_axis=pred_val_axis)


#


#
# TEST functions
#
def _test_cm():
    # test function with confusion matrix done
    array = np.array([[13, 0, 1, 0, 2, 0],
                      [0, 50, 2, 0, 10, 0],
                      [0, 13, 16, 0, 0, 3],
                      [0, 0, 0, 13, 1, 0],
                      [0, 40, 0, 1, 15, 0],
                      [0, 0, 0, 0, 0, 20]])
    # get pandas dataframe
    df_cm = DataFrame(array, index=range(1, 7), columns=range(1, 7))
    # colormap: see this and choose your more dear
    cmap = 'PuRd'
    pretty_plot_confusion_matrix(df_cm, cmap=cmap)


#

def _test_data_class():
    """ test function with y_test (actual values) and predictions (predic) """
    # data
    y_test = np.array(
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2,
         3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4,
         5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    predic = np.array(
        [1, 2, 4, 3, 5, 1, 2, 4, 3, 5, 1, 2, 3, 4, 4, 1, 4, 3, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2,
         4, 4, 5, 1, 2, 3, 3, 5, 1, 2, 3, 3, 5, 1, 2, 3, 4, 4, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 1, 2, 4, 4,
         5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 4, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    """
      Examples to validate output (confusion matrix plot)
        actual: 5 and prediction 1   >>  3
        actual: 2 and prediction 4   >>  1
        actual: 3 and prediction 4   >>  10
    """

    columns = []
    annot = True;
    cmap = 'GnBu';
    fmt = '.2f'
    lw = 0.5
    cbar = False
    show_null_values = 2
    pred_val_axis = 'y'
    # size::
    fz = 12;
    figsize = [9, 9];
    if (len(y_test) > 10):
        fz = 9;
        figsize = [14, 14];
    plot_confusion_matrix_from_data(y_test, predic, columns,
                                    annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)


#


#
# MAIN function
#
if (__name__ == '__main__'):
    print('__main__')
    print('_test_cm: test function with confusion matrix done\nand pause')
    # _test_cm()
    plt.pause(5)
    # print('_test_data_class: test function with y_test (actual values) and predictions (predic)')
    _test_data_class()
