"""
    Profile description
"""
##########################################################
# %% imports
##########################################################
import numpy as np
import cv2

import torch
from torch import cuda
from torch.autograd import Variable

from util.data_processing import get_inference_dataset
from util.helpers import load_brute

##########################################################
# %% DataLoader and Co
##########################################################
datadir = '../DataSets/'
outdir = '../outputs/'

### META of dataset
datadir_BBBC = datadir + 'Synthetic_Data_1'
datadir_Horvarth = datadir + 'Peter_Horvath_Subsample'
datadir_Chaffer = datadir + 'Chaffer_Data'
dataset_path = datadir_BBBC

model_name = 'Model_name_string'
path_to_GT = datadir + 'MetaData1_GT_link_CP.csv'
model_path = outdir + '2stage_infoVAE_2020-09-17-23:21_100'

n = 10  # figure with 15x15 digits
digit_size = 64

### reload model
VAE_2 = load_brute(model_path + '/VAE_1.pth')
VAE_2.eval()

### META of training deivice
device = torch.device('cpu' if not cuda.is_available() else 'cuda')
print(f'\tTrain on: {device}\t')

### load data
_, infer_dataloader = get_inference_dataset(dataset_path, 2, digit_size, shuffle=True, droplast=False)
data, _, _ = next(iter(infer_dataloader))
# start and end images
data = Variable(data, requires_grad=False).to(device)

### Encoder images
latentStart, latentEnd = VAE_2.get_latent(data)
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
recon = VAE_2.decode(list_Z)
reconstructions = torch.sigmoid(torch.squeeze(recon)).detach().cpu().permute(0, 2, 3, 1)
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

cv2.imshow("Interpolation in Image Space vs Latent Space", result)
cv2.waitKey()
cv2.destroyAllWindows()
