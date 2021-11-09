"""
    Profile description
"""
##########################################################
# %% imports
##########################################################
import numpy as np
import cv2, os

from torch.autograd import Variable

import torch

from models.networks_refactoring import twoStageBetaVAE
from models.train_net import VAEXperiment
from config import args, dataset_lookUp, device
from util.data_processing import get_inference_dataset

##########################################################
# %% config of the experimental parameters
##########################################################
# specific argument for this model
args.add_argument('--zdim1', dest="hidden_dim_aux", type=float, default=100)
args.add_argument('--alpha', type=float, default=1)
args.add_argument('--beta', type=float, default=1)
args.add_argument('--gamma', type=float, default=1)
args.add_argument('--steps', type=int, help="The number of steps in interpolation.", default=10)

# args.add_argument('--pretrained', dest='weight_path', type=str,
#                   default='/mnt/Linux_Storage/outputs/2_dsprite/pretrainDsprite_twoStageVaDE/logs/last.ckpt')
args = args.parse_args()

dataset_path = os.path.join(args.data_path, dataset_lookUp[args.dataset]['path'])
GT_path = os.path.join(args.data_path, dataset_lookUp[args.dataset]['meta'])
### load trained models if there is any ###
# if in training model, create new folder,otherwise use a existing parent directory of the pretrained weights
print(f"your model will be saved at {args.saved_model_path}")
##########################################################
# %% DataLoader and Co
##########################################################

steps = args.steps  # figure with 15x15 digits
digit_size = 64

##########################################################
# %% Load Model
##########################################################
print("============================================================")
print("====================== Set up Model ========================")
model = twoStageBetaVAE(zdim_1=args.hidden_dim_aux, zdim_2=args.hidden_dim, input_channels=args.input_channel,
                        input_size=args.input_size, alpha=args.alpha,
                        beta=args.beta, gamma=args.gamma)
Experiment = VAEXperiment(model, {
    "lr": args.learning_rate,
    "weight_decay": args.weight_decay,
    "scheduler_gamma": args.scheduler_gamma
}, log_path=args.output_path)
Experiment.load_weights(args.saved_model_path)
model.to(device)
print(f"--> Training from previous checkpoint {args.saved_model_path}!! ")

########################### load data ###########################
## randomly getting 2 images (start and end)
_, infer_dataloader = get_inference_dataset(dataset_path, 2, args.input_size, shuffle=True, droplast=False)
data, _, _ = next(iter(infer_dataloader))
data = Variable(torch.tensor(data, dtype=torch.float), requires_grad=False).to(device)

print("\n====================== Finished Setup ======================")
print("============================================================")
##########################################################
# %% Visualize latent space and save it
##########################################################
print("\n============================================================")
print("====================== Interpolation =======================")
latentStart, latentEnd = model.encode_2(data)
# images back to CPU
startImage, endImage = data.detach().cpu()
# display manifold of the images
figure = np.zeros((digit_size * steps, digit_size, 3))
alphaValues = np.linspace(0, 1, steps)
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
# img_grid = make_grid(recon[:, :3, :, :], nrow=steps, padding=12, pad_value=1)
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
    interpolatedImage = np.transpose(interpolatedImage.astype(np.uint8), (1, 2, 0))
    resultImage = interpolatedImage if resultImage is None else np.hstack([resultImage, interpolatedImage])

    reconstructedImage = reconstructions[i] * 255.
    reconstructedImage = reconstructedImage.astype(np.uint8)
    resultLatent = reconstructedImage if resultLatent is None else np.hstack([resultLatent, reconstructedImage])
    result = np.vstack([resultImage, resultLatent])

cv2.imwrite(os.path.join(args.output_path, "interpolation.jpg"), result)
cv2.imshow("Interpolation in Image Space vs Latent Space", result)
cv2.waitKey()
cv2.destroyAllWindows()
print(f"--> your image saved as {args.output_path}" + "interpolation.jpg")
print("\n=================== Finished Interpolation ===================")
print("============================================================")