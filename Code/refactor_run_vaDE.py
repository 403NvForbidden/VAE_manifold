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

from models.networks_refactoring import betaVAE, infoMaxVAE
from models.train_net import VAEXperiment
from quantitative_metrics.performance_metrics_single import compute_perf_metrics
from util.config import args, dataset_lookUp, device
from util.data_processing import get_train_val_dataloader, get_inference_dataset
from util.helpers import metadata_latent_space, plot_from_csv, get_raw_data, single_reconstruciton

##########################################################
# %% config of the experimental parameters
##########################################################
# specific argument for this model
args.add_argument('--model', default='infoMaxVAE')
args.add_argument('--alpha', type=float, default=1)
args.add_argument('--beta', type=float, default=10)
args.add_argument('--pretrained', dest='weight_path', type=str,
                  default='../outputs/infoMaxVAE_2021-02-18-01:21/logs/last.ckpt')
args = args.parse_args()
# TODO: overwrite the parameters

dataset_path = os.path.join(args.input_path, dataset_lookUp[args.dataset]['path'])
GT_path = os.path.join(args.input_path, dataset_lookUp[args.dataset]['meta'])

# if in training model, create new folder,otherwise use a existing parent directory of the pretrained weights
if args.train and args.weight_path == '':
    save_model_path = os.path.join(args.output_path,
                                   args.model + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")))
else:
    save_model_path = ('/').join(args.weight_path.split('/')[:-2])

##########################################################
# %% Train
##########################################################
### pretrain model


train_loader, valid_loader = get_train_val_dataloader(dataset_path, input_size=args.input_size,
                                                      batchsize=args.batch_size, test_split=0.05)

model = infoMaxVAE(zdim=args.hidden_dim, input_channels=args.input_channel, input_size=args.input_size,
                   alpha=args.alpha,
                   beta=args.beta)
Experiment = VAEXperiment(model, {
    "lr": args.learning_rate,
    "weight_decay": args.weight_decay,
    "scheduler_gamma": args.scheduler_gamma
}, log_path=save_model_path)
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
                         gpus=(1 if device.type == 'cuda' else 0),  # checkpoint_callback=False,
                         check_val_every_n_epoch=2, profiler='simple', callbacks=[checkpoint_callback])

    trainer.fit(Experiment, train_loader, valid_loader)