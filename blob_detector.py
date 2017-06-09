import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib
from skimage.io import imread
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

frame=61
image_gray = image = imread("embeddings/breakoutA/breakout{}.png".format(frame),as_gray=True)
# skimage.feature.blob_dog(data.coins(), threshold=.5, max_sigma=40)

blobs_log = blob_log(image_gray, max_sigma=5, num_sigma=10, threshold=.05)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

blobs_dog = blob_dog(image_gray, max_sigma=5, threshold=.001)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_doh = blob_doh(image_gray, max_sigma=5, threshold=.0001)

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)


fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
ax = axes.ravel()

for idx, (blobs, color, title) in enumerate(sequence):
    ax[idx].set_title(title)
    ax[idx].imshow(image, interpolation='nearest')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax[idx].add_patch(c)
    ax[idx].set_axis_off()

plt.tight_layout()
matplotlib.rcParams.update({'font.size': 32})
plt.show()
