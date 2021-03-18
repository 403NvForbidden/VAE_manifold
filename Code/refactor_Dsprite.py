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

from models.networks_refactoring import betaVAE, VaDE, EnhancedVAE, twoStageVaDE
from models.train_net import VAEXperiment, pretrain_vaDE_model, pretrain_vaDE_model_SSIM, \
    pretrain_EnhancedVAE_model_SSIM, pretrain_2stageVaDE_model_SSIM
from quantitative_metrics.performance_metrics_single import compute_perf_metrics
from util.Process_benchmarkDataset import get_dsprites_train_loader, get_dsprites_inference_loader
from util.config import args, dataset_lookUp, device
from util.data_processing import get_train_val_dataloader, get_inference_dataset, imshow_tensor
from util.helpers import metadata_latent_space, plot_from_csv, get_raw_data, single_reconstruciton, double_reconstruciton
from timeit import default_timer as timer

##########################################################
# %% config of the experimental parameters
##########################################################
# specific argument for this model
args.add_argument('--model', default='pretrainDsprite_twoStageVaDE')
args.add_argument('--zdim1', dest="hidden_dim_aux", type=float, default=100)
args.add_argument('--alpha', type=float, default=10)
args.add_argument('--beta', type=float, default=1)
args.add_argument('--gamma', type=float, default=10)
args.add_argument('--pretrained', dest='weight_path', type=str,
                  default='')
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

##########################################################
# %% Train
##########################################################
if args.train: train_loader, valid_loader = get_dsprites_train_loader(batch_size=args.batch_size, val_split=0.05)

# model = EnhancedVAE(zdim=args.hidden_dim, input_channels=args.input_channel, input_size=args.input_size, beta=args.beta)
# model = VaDE(zdim=args.hidden_dim, ydim=4, input_channels=args.input_channel, input_size=args.input_size)
model = twoStageVaDE(zdim_1=args.hidden_dim_aux, zdim_2=args.hidden_dim, input_channels=args.input_channel,
                     input_size=args.input_size, alpha=args.alpha,
                     beta=args.beta, gamma=args.gamma, ydim=3)

Experiment = VAEXperiment(model, {
    "lr": args.learning_rate,
    "weight_decay": args.weight_decay,
    "scheduler_gamma": args.scheduler_gamma
}, log_path=save_model_path)

### pretrain
# if args.train: pretrain_vaDE_model_SSIM(model, train_loader, pre_epoch=40, save_path=save_model_path, device=device)
# pretrain_EnhancedVAE_model_SSIM(model, train_loader, pre_epoch=15, save_path=save_model_path, device=device)
if args.train: pretrain_2stageVaDE_model_SSIM(model, train_loader, pre_epoch=80, save_path=save_model_path, device=device)

Experiment.load_weights(args.weight_path)

# define the logger to log training output, the default is using tensorBoard
logger = pl_loggers.TensorBoardLogger(f'{save_model_path}/logs/', name=args.model)

# TODO: add the meaning of each arguments of Trainer
checkpoint_callback = ModelCheckpoint(
    # monitor='loss',
    dirpath=os.path.join(save_model_path, "logs"),
    # filename='ckpt-{epoch:02d}',
    save_last=True,
    # save_top_k=1,
    # mode='min',
)
if args.train and args.weight_path == '':
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, auto_scale_batch_size='binsearch', auto_lr_find=True,
                         gpus=(1 if device.type == 'cuda' else 0),  # checkpoint_callback=[checkpoint_callback],
                         profiler='simple', callbacks=[checkpoint_callback])
    trainer.fit(Experiment, train_loader, valid_loader)

##########################################################
# %% Evaluate
##########################################################

## step 1 ##
## transform the imgs in the dataset to latent dimensions
# prepare the inference dataset
_, infer_dataloader = get_dsprites_inference_loader(batch_size=512, shuffle=True)

try:
    metadata_csv = pd.read_csv(os.path.join(save_model_path, 'embeded_data.csv'), index_col=False)
    metadata_csv.dropna(inplace=True)
except:
    ## running for the first time
    metadata_csv = metadata_latent_space(model, dataloader=infer_dataloader, device=device,
                                         GT_csv_path=GT_path, save_csv=True, with_rawdata=False,
                                         csv_path=os.path.join(save_model_path, 'embeded_data.csv'))
    metadata_csv.dropna(inplace=True)
    ###########################
    ### embedding projector ###
    #####################v#####
    ### compute for tensorboard embedding projector
    embeddings = metadata_csv[[col for col in metadata_csv.columns if 'z' in col]].values
    label_list = metadata_csv.GT_label.astype(str).to_list()
    imgs = get_raw_data(infer_dataloader, metadata_csv)
    # resize these images
    if imgs.shape[-1] != 64:
        imgs = torch.Tensor(
            [np.transpose(resize(img.permute(1, 2, 0), (16, 16), preserve_range=False, anti_aliasing=True)) for img in
             imgs])
    logger.experiment.add_embedding(embeddings, label_list, label_img=imgs[:, 0:3, :, :])

    ### plotly embedding projector
    figplotly = plot_from_csv(metadata_csv, low_dim_names=['z0', 'z1', 'z2'], GT_col='GT_label', dim=3, as_str=True)
    plotly.offline.plot(figplotly, filename=os.path.join(save_model_path, 'Representation.html'), auto_open=False)

    ### Recon and Generation #####
    double_reconstruciton(infer_dataloader, model, save_model_path, device, num_img=12, logger=logger)

embeddings = metadata_csv[[col for col in metadata_csv.columns if 'z' in col]].values
label_list = metadata_csv.GT_label.astype(str).to_list()
imgs = get_raw_data(infer_dataloader, metadata_csv)
if imgs.shape[-1] != 16:
    imgs = torch.Tensor(
        [np.transpose(resize(img.permute(1, 2, 0), (25, 25), preserve_range=False, anti_aliasing=True)) for img in
         imgs])
# resize these images
# if imgs.shape[-1] != 64:
#     imgs = torch.Tensor(
#         [np.transpose(resize(img.permute(1, 2, 0), (64, 64), preserve_range=False, anti_aliasing=True)) for img in
#          imgs])
logger.experiment.add_embedding(embeddings, label_list, label_img=imgs[:, 0:3, :, :])
'''
###############################
# %% Run performance matrics ###
###############################
params_preferences = {
    'feature_size': args.input_size ** 2 * args.input_channel,
    'path_to_raw_data': dataset_path,
    # 'path_to_raw_data': '../DataSets/Selected_Hovarth',
    'dataset_tag': 1,  # 1:BBBC 2:Horvath 3:Chaffer
    'low_dim_names': ['z0', 'z1', 'z2'],
    'global_saving_path': save_model_path + '/',  # Different for each model, this one is update during optimization

    ### Unsupervised metrics
    'save_unsupervised_metric': False,
    'only_local_Q': False,
    'kt': 100,
    'ks': 300,

    ### Mutual Information
    'save_mine_metric': False,
    'batch_size': 256,
    'bound_type': 'interpolated',
    'alpha_logit': -4.6,  # result in alpha = 0.01
    'epochs': 10,

    ### Classifier accuracy
    'save_classifier_metric': False,
    'num_iteration': 3,

    ### BackBone Metric
    'save_backbone_metric': True,

    ### Disentanglement Metric
    'save_disentanglement_metric': False,
    'features': dataset_lookUp[args.dataset]['feat'],
}
compute_perf_metrics(metadata_csv, params_preferences, logger)
# finally close the logger
'''
logger.close()
