import numpy as np
import skimage.color
import scipy.signal
import tsne

from matplotlib import pyplot as plot
from PIL import Image

image = Image.open('ver.jpg')
v_rgb = np.array(image, dtype=np.float32) / 255.0
v_lab = skimage.color.rgb2lab(v_rgb)
v_mean = np.mean(v_lab, axis=(0, 1), keepdims=True)
v_dist = np.sqrt(np.sum(np.square(v_lab - v_mean), axis=2))


def save(v):
    v = np.uint8(255 * v / np.max(v))
    image = Image.fromarray(v)
    image.save('save.png')


def f0():
    (x, y) = np.meshgrid(np.linspace(0, 1, v_lab.shape[1]), np.linspace(0, 1, v_lab.shape[0]))
    f = np.concatenate([v_lab, x[..., None], y[..., None]], 2)
    f = (f - np.min(f, axis=(0, 1))) / (np.max(f, axis=(0, 1)) - np.min(f, axis=(0, 1)))
    t = tsne.bh_sne(f.reshape([-1, 5]))

    plot.figure(figsize=(8, 8))
    plot.scatter(t[:, 0], t[:, 1], s=3, c=v_rgb.reshape([-1, 3]))
    plot.savefig('0.png')


def f1():
    v_feat = np.mean(v_dist, axis=0)

    plot.figure(figsize=(8, 12))
    plot.subplot(3, 1, 1)
    plot.imshow(v_rgb)
    plot.subplot(3, 1, 2)
    plot.imshow(v_dist, cmap='gray')
    plot.subplot(3, 1, 3)
    plot.plot(v_feat)
    plot.xlim([0, 200])
    plot.savefig('1.png')

def f2():
    k = np.ones((15, 15))
    v_patch = scipy.signal.convolve2d(v_dist, k, mode='same')

    plot.figure(figsize=(8, 12))
    plot.subplot(3, 1, 1)
    plot.imshow(v_rgb)
    plot.subplot(3, 1, 2)
    plot.imshow(v_dist, cmap='gray')
    plot.subplot(3, 1, 3)
    plot.imshow(v_patch, cmap='gray')
    plot.savefig('2.png')
