# @Author: Sacha Haidinger <sachahai>
# @Date:   2020-04-13T10:26:28+10:00
# @Email:  sacha.haidinger@epfl.ch
# @Project: Learning methods for Cell Profiling
# @Last modified by:   sachahai
# @Last modified time: 2020-04-13T16:01:26+10:00

import numpy as np
from skimage import io
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

file = f'single_cell_processed/Wella1_Slide1_Condition1_XY1/cell1.tif'

img = io.imread(file)
fig = plt.figure()
plt.imshow(img[:,:,:3])
plt.show(block=True)


np.min(img)
np.max(img)
img.dtype


from skimage.util import img_as_int, img_as_float


img.dtype
img_new = img.astype('int16',casting='safe')
img_new.dtype
np.min(img_new)
np.max(img_new)

img_stretch = exposure.rescale_intensity(img_new, in_range='uint8',out_range=(-2**15,2**15 -1))
img_stretch.dtype
np.min(img_stretch)
plt.imshow(img_stretch[:,:,0])

imgfinal = img_as_float(img_stretch)

imgfinal.dtype
plt.imshow(imgfinal)
np.min(imgfinal)
np.max(imgfinal)
plt.imshow(imgfinal,vmin=-1,vmax=1)
