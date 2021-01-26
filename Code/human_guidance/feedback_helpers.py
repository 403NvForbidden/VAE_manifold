# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-08-29T15:18:03+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-08-29T23:52:05+10:00

'''
Miscellaneous functions and classess that were implemented to propose a first simple
framework to give human feedback to VAE based on an additional term in its objective function

This work is meant to be a 'proof of concept'; Feedback can be incorporated in VAE weights as long
as it can be translated as an additionnal term inside the objective function.
We studied here a very simple case with Google Dsprites dataset.
The feedback is in the form of anchored several points to a given region of the latent space.
That is perform with a MSE addtional term, weighted by a parameters that is sample depended, meaning
that it is set to 0 for all samples except the one that are constrained. This information need to be
stored in a CSV file.

DSprites Dataset from [1].
----------
[1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick,
    M., ... & Lerchner, A. (2017). beta-vae: Learning basic visual concepts
    with a constrained variational framework. In International Conference
    on Learning Representations.
'''
import numpy as np
import pandas as pd
import datetime
import itertools
import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


#######################################
### Helper function provided by Google
#######################################
def latent_to_index(latents):
    metadata = {
        'description': 'Disentanglement test Sprites dataset.Procedurally generated 2D shapes, from 6 disentangled latent factors.This dataset uses 6 latents, controlling the color, shape, scale, rotation and position of a sprite. All possible variations of the latents are present. Ordering along dimension 1 is fixed and can be mapped back to the exact latent values that generated that image.We made sure that the pixel outputs are different. No noise added.',
        'latents_sizes': np.array([1,
                                   3, 6, 40, 32, 32]),
        'latents_names': ('color', 'shape', 'scale', 'orientation', 'posX', 'posY'), 'date': 'April 2017', 'version': 1,
        'title': 'dSprites dataset',
        'latents_possible_values': {'posX': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                                      0.16129032, 0.19354839, 0.22580645, 0.25806452, 0.29032258,
                                                      0.32258065, 0.35483871, 0.38709677, 0.41935484, 0.4516129,
                                                      0.48387097, 0.51612903, 0.5483871, 0.58064516, 0.61290323,
                                                      0.64516129, 0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                                      0.80645161, 0.83870968, 0.87096774, 0.90322581, 0.93548387,
                                                      0.96774194, 1.]),
                                    'posY': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                                      0.16129032, 0.19354839, 0.22580645, 0.25806452, 0.29032258,
                                                      0.32258065, 0.35483871, 0.38709677, 0.41935484, 0.4516129,
                                                      0.48387097, 0.51612903, 0.5483871, 0.58064516, 0.61290323,
                                                      0.64516129, 0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                                      0.80645161, 0.83870968, 0.87096774, 0.90322581, 0.93548387,
                                                      0.96774194, 1.]),
                                    'scale': np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
                                    'orientation': np.array([0., 0.16110732, 0.32221463, 0.48332195, 0.64442926,
                                                             0.80553658, 0.96664389, 1.12775121, 1.28885852, 1.44996584,
                                                             1.61107316, 1.77218047, 1.93328779, 2.0943951, 2.25550242,
                                                             2.41660973, 2.57771705, 2.73882436, 2.89993168, 3.061039,
                                                             3.22214631, 3.38325363, 3.54436094, 3.70546826, 3.86657557,
                                                             4.02768289, 4.1887902, 4.34989752, 4.51100484, 4.67211215,
                                                             4.83321947, 4.99432678, 5.1554341, 5.31654141, 5.47764873,
                                                             5.63875604, 5.79986336, 5.96097068, 6.12207799,
                                                             6.28318531]), 'shape': np.array([1., 2., 3.]),
                                    'color': np.array([1.])}, 'author': 'lmatthey@google.com'}
    latents_sizes = metadata['latents_sizes']
    latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                    np.array([1, ])))
    return np.dot(latents, latents_bases).astype(int)


def sample_latent(size=1):
    metadata = {
        'description': 'Disentanglement test Sprites dataset.Procedurally generated 2D shapes, from 6 disentangled latent factors.This dataset uses 6 latents, controlling the color, shape, scale, rotation and position of a sprite. All possible variations of the latents are present. Ordering along dimension 1 is fixed and can be mapped back to the exact latent values that generated that image.We made sure that the pixel outputs are different. No noise added.',
        'latents_sizes': np.array([1,
                                   3, 6, 40, 32, 32])}
    latents_sizes = metadata['latents_sizes']
    samples = np.zeros((size, latents_sizes.size))
    for lat_i, lat_size in enumerate(latents_sizes):
        samples[:, lat_i] = np.random.randint(lat_size, size=size)
    return samples


# Helper function to show images
def show_images_grid(imgs_, num_images=16):
    ncols = int(np.ceil(num_images ** 0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()
    rand_ind = np.random.randint(low=0, high=imgs_.shape[0], size=num_images)
    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs_[rand_ind[ax_i]], cmap='Greys_r', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')


def show_density(imgs):
    _, ax = plt.subplots()
    ax.imshow(imgs.mean(axis=0), interpolation='nearest', cmap='Greys_r')
    ax.grid('off')
    ax.set_xticks([])
    ax.set_yticks([])


#######################################
### Custom function and Dataset
#######################################
class DSpritesDataset(Dataset):
    """
    Custom DataLoader that loads DSprites samples by batch
    Custom DataLoader that allows to link each samples with a (potential) feedback written in a csv file

    Params of the constructor :
      imgs_subsample (ndarray,uint8) : Subsample of the image provided in DSprite dataset. N x 64 x 64
      GT_factors_subsample (ndarray,float64) : Subsample of the latent factors of each samples.
              N x 6. Named 'latents_values' in the original dataset
      Unique_id_subsample (ndarray,int) : In original dataset was generated procedurally and is ordered
              Each sample can therefore easily be mapped to a Unique index. If a subsample is generated
              this input should contain the IDs (from 0 to 737199 (size of total dataet)) of each sample.
      feedback_csv (string or None) : Path to a csv file that contains expert Feedback. This csv file should
              contain at least 4 columns : 'Unique_id' the index of each sample of the considered dataset,
              'delta' the weight of the feedback term in objective function (0 if point is not constrained),
              'x_anchor' x coordinate of the anchor point for each sample,
              'y_anchor' y coordinate of the anchor point for each sample

    """
    lat_names = ('shape', 'scale', 'orientation', 'posX', 'posY')
    lat_sizes = np.array([3, 6, 40, 32, 32])
    img_size = (1, 64, 64)
    lat_values = {'posX': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                    0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                    0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                    0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                    0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                    0.93548387, 0.96774194, 1.]),
                  'posY': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                    0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                    0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                    0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                    0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                    0.93548387, 0.96774194, 1.]),
                  'scale': np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
                  'orientation': np.array([0., 0.16110732, 0.32221463, 0.48332195,
                                           0.64442926, 0.80553658, 0.96664389, 1.12775121,
                                           1.28885852, 1.44996584, 1.61107316, 1.77218047,
                                           1.93328779, 2.0943951, 2.25550242, 2.41660973,
                                           2.57771705, 2.73882436, 2.89993168, 3.061039,
                                           3.22214631, 3.38325363, 3.54436094, 3.70546826,
                                           3.86657557, 4.02768289, 4.1887902, 4.34989752,
                                           4.51100484, 4.67211215, 4.83321947, 4.99432678,
                                           5.1554341, 5.31654141, 5.47764873, 5.63875604,
                                           5.79986336, 5.96097068, 6.12207799, 6.28318531]),
                  'shape': np.array([1., 2., 3.]),
                  'color': np.array([1.])}

    def __init__(self, imgs_subsample, GT_factors_subsample, Unique_id_subsample, feedback_csv='None',
                 transform=[transforms.ToTensor()]):

        self.imgs = imgs_subsample
        self.lat_values = GT_factors_subsample
        self.transforms = transforms.Compose(transform)
        self.unique_id = Unique_id_subsample

        self.feedback_df = feedback_csv
        if feedback_csv != 'None':  # None : No feedback was given yet
            self.feedback_df = pd.read_csv(feedback_csv)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        # stored image have binary and shape (H x W) so multiply by 255 to get pixel
        # values + add dimension
        sample = np.expand_dims(self.imgs[idx] * 255, axis=-1)

        sample = self.transforms(sample)

        lat_value = self.lat_values[idx]
        unique_id = self.unique_id[idx]

        if isinstance(self.feedback_df, str):  # No feedback yet
            delta_param = np.zeros(self.imgs.shape[0])
            x_anchor = np.zeros((self.imgs.shape[0], 1))
            y_anchor = np.zeros((self.imgs.shape[0], 1))
            delta = delta_param[idx]
            x_anch = x_anchor[idx]
            y_anch = y_anchor[idx]
        else:
            sub_df = self.feedback_df.loc[self.feedback_df['Unique_id'].isin([unique_id])]
            delta = sub_df.delta.values
            x_anch = sub_df.x_anchor.values
            y_anch = sub_df.y_anchor.values

        return sample, lat_value, unique_id, (delta, x_anch, y_anch)


def save_latent_code(model, dataloader, save_path='noname_metadata.csv', train_on_gpu=True):
    '''
    From a trained model, evaluate the latent code on the desired subsample of DSprites dataset
    and safe in a CSV file the Unique index of each sample as well as the latent code.
    '''
    unique_id_list = []
    shape_list = []
    scale_list = []
    orientation_list = []
    posX_list = []
    posY_list = []
    z_list = []
    for i, (data, labels, unique_id, _) in enumerate(dataloader):
        # Extract unique cell id from file_names
        unique_id_list.append(unique_id.numpy())
        shape_list.append([labels[i, 1].numpy().item() for i in range(labels.shape[0])])
        scale_list.append([labels[i, 2].numpy().item() for i in range(labels.shape[0])])
        orientation_list.append([labels[i, 3].numpy().item() for i in range(labels.shape[0])])
        posX_list.append([labels[i, 4].numpy().item() for i in range(labels.shape[0])])
        posY_list.append([labels[i, 5].numpy().item() for i in range(labels.shape[0])])

        if train_on_gpu:
            # make sure this lives on the GPU
            data = data.cuda()
        with torch.no_grad():
            model.eval()

            data = Variable(data, requires_grad=False)

            z, _ = model.encode(data)
            z = z.view(-1, model.zdim)
            z_list.append((z.data).cpu().numpy())
    unique_id_list = list(itertools.chain.from_iterable(unique_id_list))
    shape_list = list(itertools.chain.from_iterable(shape_list))
    scale_list = list(itertools.chain.from_iterable(scale_list))
    orientation_list = list(itertools.chain.from_iterable(orientation_list))
    posX_list = list(itertools.chain.from_iterable(posX_list))
    posY_list = list(itertools.chain.from_iterable(posY_list))
    z_points = np.concatenate(z_list, axis=0)
    df = pd.DataFrame(
        columns=['Unique_id', 'shape', 'scale', 'orientation', 'posX', 'posY', 'VAE_x_coord', 'VAE_y_coord'])
    df.VAE_x_coord = z_points[:, 0]
    df.VAE_y_coord = z_points[:, 1]
    df.Unique_id = unique_id_list
    df.shape = shape_list
    df.scale = scale_list
    df.orientation = orientation_list
    df.posX = posX_list
    df.posY = posY_list

    df.to_csv(save_path)

    return df


def plot_train_history(history):
    '''
    Plot the training history of a simple VAE model that has been trained

    Params :
        history (pandas Dataframe) : Dataframe containing the training history
    '''
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 1, 1)  # Frist full row
    ax2 = fig.add_subplot(2, 2, 3)  # bottom left on 4x4 grid
    ax3 = fig.add_subplot(2, 2, 4)  # bottom right on a 4x4 grid
    ax1.plot(history['global_VAE_loss'], color='dodgerblue', label='train')
    ax1.set_title('Global VAE Loss')
    ax2.plot(history['recon_loss'], color='dodgerblue', label='train')
    ax2.set_title('Reconstruction Loss')
    ax3.plot(history['kl_loss'], color='dodgerblue', label='train')
    ax3.set_title('Fit to Prior')
    ax1.legend()
    ax2.legend()
    ax3.legend()

    return fig
