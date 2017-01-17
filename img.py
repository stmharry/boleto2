import numpy as np
import skimage.color

from matplotlib import pyplot as plot
from PIL import Image

image = Image.open('img.jpg')

x_rgb = np.array(image, dtype=np.float32) / 255.0
x_lab = skimage.color.rgb2lab(x_rgb)
x_mean = np.mean(x_lab, axis=(0, 1), keepdims=True)
x_dist = np.sqrt(np.sum(np.square(x_lab - x_mean), axis=2))
x_feat = np.mean(x_dist, axis=0)

plot.figure(figsize=(4, 6))
plot.subplot(3, 1, 1)
plot.imshow(x_rgb)
plot.subplot(3, 1, 2)
plot.imshow(x_dist, cmap='gray')
plot.subplot(3, 1, 3)
plot.plot(x_feat)
plot.savefig('fig.png')
