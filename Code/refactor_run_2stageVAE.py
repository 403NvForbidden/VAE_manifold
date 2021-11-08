##########################################################
# %% import packages
##########################################################
import datetime
import os
import pandas as pd

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import plotly.offline

from models.networks_refactoring import twoStageInfoMaxVAE, twoStageVaDE, twoStageBetaVAE, infoMaxVAE, VaDE
from models.train_net import VAEXperiment, pretrain_2stageVaDE_model_SSIM, pretrain_2stageVAEmodel_SSIM
from quantitative_metrics.classifier_metric import dsprite_classifier_performance
from quantitative_metrics.performance_metrics_single import compute_perf_metrics
from quantitative_metrics.unsupervised_metric import save_representation_plot
from util.config import args, dataset_lookUp, device
from util.data_processing import get_train_val_dataloader, get_inference_dataset
from torchsummary import summary
from util.helpers import metadata_latent_space, plot_from_csv, get_raw_data, double_reconstruciton, make_path
from pprint import pprint

##########################################################
# %% config of the experimental parameters
##########################################################
###  argument for this model
args.add_argument('--model', type=str, help="The name of the model",
                  default="2StageVaDE " + str(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")))
args.add_argument('--zdim1', dest="hidden_dim_aux", type=float, default=100)
args.add_argument('--alpha', type=float, default=1)
args.add_argument('--beta', type=float, default=1)
args.add_argument('--gamma', type=float, default=1)
args.add_argument('--num_pretrain', type=int, default=2)
args.add_argument('--ydim', help="The number of Gaussian Models.", type=int, default=3)
args.add_argument('--pretrained', dest='pretrained_path', help="Load Pretrained Model from path", type=str, default='')
args = args.parse_args()

dataset_path = os.path.join(args.data_path, dataset_lookUp[args.dataset]['path'])
GT_path = os.path.join(args.data_path, dataset_lookUp[args.dataset]['meta'])
### load trained models if there is any ###
# if in training model, create new folder,otherwise use a existing parent directory of the pretrained weights
if args.train and args.pretrained_path == '' and args.saved_model_path == '':
    save_model_path = os.path.join(args.output_path, args.model)
elif args.saved_model_path != '':
    save_model_path = ('/').join(args.pretrained_path.split('/')[:-1])
elif args.pretrained_path != '':
    save_model_path = args.pretrained_path

print(f"your model will be saved at {save_model_path}")
##########################################################
# %% Train
##########################################################
print("============================================================")
print("===================Set up training Env======================")
print("============================================================")
train_loader, valid_loader = get_train_val_dataloader(dataset_path, input_size=args.input_size,
                                                      batchsize=args.batch_size, test_split=0.1)
model = twoStageVaDE(zdim_1=args.hidden_dim_aux, zdim_2=args.hidden_dim, input_channels=args.input_channel,
                     input_size=args.input_size, alpha=args.alpha,
                     beta=args.beta, gamma=args.gamma, ydim=args.ydim)
# model = infoMaxVAE(zdim=args.hidden_dim, input_channels=args.input_channel, input_size=args.input_size,
#                    alpha=args.alpha,
#                    beta=args.beta)
# model = VaDE(zdim=args.hidden_dim, dim=args.ydim, input_channels=args.input_channel, input_size=args.input_size)

Experiment = VAEXperiment(model, {
    "lr": args.learning_rate,
    "weight_decay": args.weight_decay,
    "scheduler_gamma": args.scheduler_gamma
}, log_path=save_model_path)
# define the logger to log training output, the default is using tensorBoard
logger = pl_loggers.TensorBoardLogger(f'{save_model_path}/logs/', name=args.model)

print(args.train)
if args.train and args.saved_model_path == '':
    pretrain_2stageVaDE_model_SSIM(model, train_loader,
                                   pre_epoch=args.num_pretrain,
                                   save_path=save_model_path,
                                   load_path=args.pretrained_path,
                                   logger=logger, device=device)

checkpoint_callback = ModelCheckpoint(
    # monitor='loss',
    dirpath=os.path.join(save_model_path, "logs"),
    # filename='ckpt-{epoch:02d}',
    save_last=True,
    # save_top_k=1,
    # mode='min',
)

if args.saved_model_path != '':
    print(
        f"--> Set up training Env -> Training from previous checkpoint {args.saved_model_path}!!")
    Experiment.load_weights(args.saved_model_path)

if args.train:
    print("--> Set up training Env -> Training from SCRATCH!!")
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, auto_scale_batch_size='binsearch', auto_lr_find=True,
                         gpus=(1 if device.type == 'cuda' else 0),
                         check_val_every_n_epoch=2, profiler='simple', callbacks=[checkpoint_callback])
    # call tune to find the batch size
    # trainer.tune(Experiment, train_loader)
    trainer.fit(Experiment, train_loader, valid_loader)
print("============================================================")
print("===================Finished training ======================")
print("============================================================\n")
##########################################################
# %% Evaluate
##########################################################
print("============================================================")
print("======================= Evaluating =========================")
print("============================================================")
### prepare the inference dataset
if args.eval:
    _, infer_dataloader = get_inference_dataset(dataset_path, batchsize=args.batch_size,
                                                input_size=args.input_size)
    if os.path.exists(os.path.join(save_model_path, 'embeded_data.csv')):
        print(
            f"--> Evaluating -> Directly loading csv files at {os.path.join(save_model_path, 'embeded_data.csv')}")
        metadata_csv = pd.read_csv(os.path.join(save_model_path, 'embeded_data.csv'), index_col=False)
        metadata_csv.dropna(inplace=True)
    else:
        ## running for the first time`
        metadata_csv = metadata_latent_space(model, dataloader=infer_dataloader, device=device,
                                             GT_csv_path=GT_path, save_csv=True, with_rawdata=False,
                                             csv_path=os.path.join(save_model_path, 'embeded_data.csv'))
        metadata_csv.dropna(inplace=True)
        ### embedding projector ###
        # compute for tensorboard embedding projector
        embeddings = metadata_csv[[col for col in metadata_csv.columns if 'z' in col]].values
        label_list = metadata_csv.GT_label.astype(str).to_list()
        imgs = get_raw_data(infer_dataloader, metadata_csv)
        # only select the first 3 channels
        logger.experiment.add_embedding(embeddings, label_list, label_img=imgs[:, :3, :, :])

        # plotly 3d projections
        figplotly = plot_from_csv(metadata_csv, low_dim_names=['z0', 'z1', 'z2'],
                                  GT_col=dataset_lookUp[args.dataset]['GT'], dim=3, as_str=True)
        plotly.offline.plot(figplotly, filename=os.path.join(save_model_path, 'Representation.html'), auto_open=False)
        double_reconstruciton(infer_dataloader, model, save_model_path, device, num_img=12, logger=logger)

print("============================================================")
print("=================== Finished Evaluating ====================")
print("============================================================\n")
###############################
# %% Run performance matrics ###
###############################
if args.benchmark:
    params_preferences = {
        'feature_size': args.input_size * args.input_size * args.input_channel,
        'path_to_raw_data': dataset_path,
        'dataset_tag': 1,  # 1:BBBC 2:Horvath 3:Chaffer
        'low_dim_names': ['z0', 'z1', 'z2'],
        'global_saving_path': save_model_path + '/',  # Different for each model, this one is update during optimization

        ### Unsupervised metrics
        'save_unsupervised_metric': False,
        'only_local_Q': False,
        'kt': 300,
        'ks': 500,

        ### Mutual Information
        'save_mine_metric': False,
        'batch_size': 256,
        'bound_type': 'interpolated',
        'alpha_logit': -4.6,  # result in alpha = 0.01
        'epochs': 10,

        ### Classifier accuracy
        'save_classifier_metric': True,
        'num_iteration': 3,

        ### BackBone Metric
        'save_backbone_metric': False,

        ### Disentanglement Metric
        'save_disentanglement_metric': False,
        'features': dataset_lookUp[args.dataset]['feat'],
    }
    compute_perf_metrics(metadata_csv, params_preferences, logger)
logger.close()
