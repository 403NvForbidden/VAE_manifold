##########################################################
# %% imports
##########################################################
import datetime
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import cuda

from models.networks_refactoring import twoStageVAE
from models.train_net import VAEXperiment
from util.config import args, dataset_lookUp
from util.data_processing import get_train_val_dataloader
from torchsummary import summary
##########################################################
# %% config of the experimental parameters
##########################################################
# specific argument for this model
args.add_argument('--model', default='twoStageVAE')
args.add_argument('--alpha', type=float, default=10)
args.add_argument('--beta', type=float, default=10)
args.add_argument('--gamma', type=float, default=100)
args.add_argument('--pretrained', dest='weight_path', type=str,
                  default='')
args = args.parse_args()
# TODO: overwrite the parameters

dataset_path = os.path.join(args.input_path, dataset_lookUp[args.dataset]['path'])
GT_path = os.path.join(args.input_path, dataset_lookUp[args.dataset]['meta'])

### META
device = torch.device('cpu' if not cuda.is_available() else 'cuda')

# if in training model, create new folder,otherwise use a existing parent directory of the pretrained weights
if args.train and args.weight_path == '':
    save_model_path = os.path.join(args.output_path,
                                   args.model + "_" + str(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")))
else:
    save_model_path = ('/').join(args.weight_path.split('/')[:-1])

##########################################################
# %% Train
##########################################################
train_loader, valid_loader = get_train_val_dataloader(dataset_path, input_size=args.input_size,
                                                      batchsize=args.batch_size // 2, test_split=0.1)

model = twoStageVAE(zdim_1=args.hidden_dim, zdim_2=100, input_channels=args.input_channel, input_size=args.input_size, alpha=args.alpha,
                   beta=args.beta, gamma=args.gamma)

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
    monitor='loss',
    dirpath=os.path.join(save_model_path, "logs"),
    filename='ckpt-{epoch:02d}',
    save_top_k=1,
    mode='min',
)
trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, auto_scale_batch_size='binsearch', auto_lr_find=True,
                     gpus=(1 if device.type == 'cuda' else 0),  # checkpoint_callback=False,
                     check_val_every_n_epoch=2, profiler='simple', callbacks=[checkpoint_callback])
# call tune to find the batch size
# trainer.tune(Experiment, train_loader)
if args.train:
    trainer.fit(Experiment, train_loader, valid_loader)