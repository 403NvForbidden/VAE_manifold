##########################################################
# %% imports
##########################################################
import datetime
import os
import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import plotly.offline
from skimage.transform import resize, rotate, rescale
import torchvision
import numpy as np

from models.networks_refactoring import betaVAE, VaDE, EnhancedVAE, infoMaxVAE, twoStageVaDE, twoStageInfoMaxVAE, \
    twoStageBetaVAE
from models.train_net import VAEXperiment, pretrain_vaDE_model, pretrain_vaDE_model_SSIM, \
    pretrain_EnhancedVAE_model_SSIM, pretrain_2stageVaDE_model_SSIM, pretrain_2stageVAEmodel_SSIM
from quantitative_metrics.performance_metrics_single import compute_perf_metrics
from util.config import args, dataset_lookUp, device
from util.data_processing import get_train_val_dataloader, get_inference_dataset, imshow_tensor
from util.helpers import metadata_latent_space, plot_from_csv, get_raw_data, single_reconstruciton, double_reconstruciton
from timeit import default_timer as timer

##########################################################
# %% config of the experimental parameters
##########################################################
# specific argument for this model
args.add_argument('--model', default='t-SNE')
args.add_argument('--zdim1', dest="hidden_dim_aux", type=float, default=100)
args.add_argument('--alpha', type=float, default=10)
args.add_argument('--beta', type=float, default=1)
args.add_argument('--gamma', type=float, default=10)
args.add_argument('--pretrained', dest='weight_path', type=str,
                  default='/mnt/Linux_Storage/outputs/1_Felix/t-SNE/logs/embeded_data.csv')
args = args.parse_args()
# TODO: overwrite the parameters

dataset_path = os.path.join(args.input_path, dataset_lookUp[args.dataset]['path'])
GT_path = os.path.join(args.input_path, dataset_lookUp[args.dataset]['meta'])

# if in training model, create new folder,otherwise use a existing parent directory of the pretrained weights
if args.train and args.weight_path == '':
    save_model_path = os.path.join(args.output_path,
                                   args.model)  # + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")))
else:
    save_model_path = ('/').join(args.weight_path.split('/')[:-2])


logger = pl_loggers.TensorBoardLogger(f'{save_model_path}/logs/', name=args.model)

metadata_csv = pd.read_csv(os.path.join(save_model_path, 'embeded_data.csv'), index_col=False)
###############################
# %% Run performance matrics ###
###############################
params_preferences = {
    'feature_size': args.input_size ** 2 * args.input_channel,
    'path_to_raw_data': dataset_path,
    # 'feature_size': 64*64*4,
    # 'path_to_raw_data': '../DataSets/Selected_Hovarth',
    'dataset_tag': 1,  # 1:BBBC 2:Horvath 3:Chaffer
    'low_dim_names':  ['tsne_0', 'tsne_1', 'tsne_2'], # ['umap_0', 'umap_1', 'umap_2'] , #
    'global_saving_path': save_model_path + '/',  # Different for each model, this one is update during optimization

    ### Unsupervised metrics
    'save_unsupervised_metric': True,
    'only_local_Q': False,
    'kt': 300,
    'ks': 500,

    ### Mutual Information
    'save_mine_metric': True,
    'batch_size': 256,
    'bound_type': 'interpolated',
    'alpha_logit': -4.6,  # result in alpha = 0.01
    'epochs': 10,

    ### Classifier accurac4.9y
    'save_classifier_metric': False,
    'num_iteration': 3,

    ### BackBone Metric
    'save_backbone_metric': False,

    ### Disentanglement Metric
    'save_disentanglement_metric': True,
    'features': dataset_lookUp[args.dataset]['feat'],
}
compute_perf_metrics(metadata_csv, params_preferences, logger)
# finally close the logger

logger.close()
