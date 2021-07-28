"""
    Profile description
"""
##########################################################
# %% imports
##########################################################
import numpy as np
import cv2, os

import torch
from torch import cuda
from torch.autograd import Variable

from util.Process_benchmarkDataset import get_dsprites_inference_loader
from util.data_processing import get_inference_dataset
from util.helpers import load_brute
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import plotly.offline

from models.networks_refactoring import twoStageInfoMaxVAE, twoStageVaDE, twoStageBetaVAE
from models.train_net import VAEXperiment, pretrain_2stageVaDE_model_SSIM, pretrain_2stageVAEmodel_SSIM
from quantitative_metrics.classifier_metric import dsprite_classifier_performance
from quantitative_metrics.performance_metrics_single import compute_perf_metrics
from quantitative_metrics.unsupervised_metric import save_representation_plot
from util.config import args, dataset_lookUp, device
from util.data_processing import get_train_val_dataloader, get_inference_dataset
from torchsummary import summary

##########################################################
# %% config of the experimental parameters
##########################################################
# specific argument for this model
from util.helpers import metadata_latent_space, plot_from_csv, get_raw_data, double_reconstruciton

args.add_argument('--model', default='twoStageInfoMaxVaDE')
args.add_argument('--zdim1', dest="hidden_dim_aux", type=float, default=100)
args.add_argument('--alpha', type=float, default=1)
args.add_argument('--beta', type=float, default=1)
args.add_argument('--gamma', type=float, default=1)
args.add_argument('--pretrained', dest='weight_path', type=str,
                  default='/mnt/Linux_Storage/outputs/2_dsprite/pretrainDsprite_twoStageVaDE/logs/last.ckpt')
args = args.parse_args()

##########################################################
# %% DataLoader and Co
##########################################################
datadir = '../DataSets/'
outdir = '../outputs/'
print(os.listdir())
### META of dataset
datadir_BBBC = datadir + 'Synthetic_Data_1'
dataset_path = datadir_BBBC

save_model_path = ('/').join(args.weight_path.split('/')[:-2])

n = 10  # figure with 15x15 digits
digit_size = 64

### reload model
model = twoStageBetaVAE(zdim_1=args.hidden_dim_aux, zdim_2=args.hidden_dim, input_channels=args.input_channel,
                    input_size=args.input_size, alpha=args.alpha,
                    beta=args.beta, gamma=args.gamma)
# model.load_state_dict(torch.load(args.weight_path))
Experiment = VAEXperiment(model, {
    "lr": args.learning_rate,
    "weight_decay": args.weight_decay,
    "scheduler_gamma": args.scheduler_gamma
}, log_path=save_model_path)
Experiment.load_weights(args.weight_path)
# weight loaded
print("weight loaded")
### META of training deivice
device = torch.device('cpu' if not cuda.is_available() else 'cuda')
model.to(device)
########################### load data ###########################
# BBBC
# _, infer_dataloader = get_inference_dataset(dataset_path, 2, digit_size, shuffle=True, droplast=False)
# dsprite
# _, infer_dataloader = get_dsprites_inference_loader(batch_size=2, shuffle=True)

# data, _, _ = next(iter(infer_dataloader))
# start and end images
data = np.load('/mnt/Linux_Storage/outputs/2_dsprite/rotation.npy')
data = Variable(torch.tensor(data, dtype=torch.float), requires_grad=False).to(device)

### Encoder images
latentStart, latentEnd = model.encode_2(data)
# images back to CPU
startImage, endImage = data.detach().cpu()

##########################################################
# %% Visualize latent space and save it
##########################################################
# display manifold of the images
figure = np.zeros((digit_size * n, digit_size, 3))

alphaValues = np.linspace(0, 1, n)
list_Z = []
raw_images = []
for alpha in alphaValues:
    # Latent space interpolation
    vec = latentStart * (1 - alpha) + latentEnd * alpha
    list_Z.append(vec)
    # Image space interpolation
    blendImage = cv2.addWeighted(np.float32(startImage), 1 - alpha, np.float32(endImage), alpha, 0)
    raw_images.append(blendImage)

list_Z = Variable(torch.stack([x for x in list_Z], dim=0), requires_grad=False).to(device)
recon = model.decode_aux(list_Z)
reconstructions = torch.sigmoid(recon).detach().cpu().permute(0, 2, 3, 1)
reconstructions = np.asarray(reconstructions)

# ### RGB channel
# img_grid = make_grid(recon[:, :3, :, :], nrow=n, padding=12, pad_value=1)
#
# plt.figure(figsize=(25, 25))
# plt.axis('off')
# plt.imshow(img_grid.detach().cpu().permute(1, 2, 0))
# # plt.savefig(model_path + '/ij_z0.25n.png')
# plt.show()
#
# reconstructions = decoder.predict(vectors)

# Put final image together
resultLatent = None
resultImage = None
# for j in range()
for i in range(len(reconstructions)):
    interpolatedImage = raw_images[i] * 255
    # interpolatedImage = cv2.resize(interpolatedImage, (50, 50))
    interpolatedImage = np.transpose(interpolatedImage.astype(np.uint8), (1, 2, 0))
    resultImage = interpolatedImage if resultImage is None else np.hstack([resultImage, interpolatedImage])

    reconstructedImage = reconstructions[i] * 255.
    # reconstructedImage = reconstructedImage.reshape([28, 28])
    # reconstructedImage = cv2.resize(reconstructedImage, (50, 50))
    reconstructedImage = reconstructedImage.astype(np.uint8)
    resultLatent = reconstructedImage if resultLatent is None else np.hstack([resultLatent, reconstructedImage])
    result = np.vstack([resultImage, resultLatent])

cv2.imwrite("/mnt/Linux_Storage/outputs/2_dsprite/rotation_interpolation.jpg", result)
cv2.imshow("Interpolation in Image Space vs Latent Space", result)
cv2.waitKey()
cv2.destroyAllWindows()
