import abc
import matplotlib.pyplot as plot
import numpy as np
import scipy.interpolate
import skimage.color
import skimage.filters


class Panel(object):
    def __init__(self,
                 num_rows,
                 num_cols,
                 height,
                 width):

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num = num_rows * num_cols

        self.height = height
        self.width = width

    def show(self, img, name, vmin=None, vmax=None):
        img = np.float32(img)

        plot.figure()
        img = np.reshape(img, (self.num_rows, self.num_cols, self.height, self.width, -1))
        img = np.transpose(img, (0, 2, 1, 3, 4))
        img = np.reshape(img, (self.num_rows * self.height, self.num_cols * self.width, -1))
        img = np.squeeze(img)
        plot.imsave('{}.png'.format(name), img, vmin=vmin, vmax=vmax, cmap='gray')

    def hist(self, img, name, mark=None):
        if mark is None:
            mark = np.empty((self.num, 0))

        plot.figure(figsize=(self.num_cols * self.width / 100., self.num_rows * self.height / 100.))
        for num_row in xrange(self.num_rows):
            for num_col in xrange(self.num_cols):
                num = num_row * self.num_cols + num_col
                img_ = img[num]
                mark_ = mark[num]

                ax = plot.subplot(self.num_rows, self.num_cols, num + 1)
                ax.hist(img_.flatten(), bins=128, range=(0, 100))
                for m in mark_:
                    ax.axvline(x=m, color='r', linestyle='dashed')
                ax.axis('off')
                ax.set_yscale('log', nonposy='clip')
                ax.set_xticklabels([])
                ax.set_yticklabels([])

        plot.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        plot.savefig('{}.png'.format(name))


class Processing(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.name = 'processing'

    @abc.abstractmethod
    def eval(self):
        pass

    @abc.abstractmethod
    def show(self):
        pass


class Coloring(Processing):
    def __init__(self):
        super(Coloring, self).__init__()
        self.name = 'coloring'

    def eval(self, img):
        self.img = skimage.color.rgb2lab(img)[..., 0]

    def show(self, panel, name):
        panel.show(self.img, name='{}_{}_img'.format(name, self.name), vmin=0, vmax=100)


class Thresholding(Processing):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(Thresholding, self).__init__()
        self.name = 'thresholding'

    @abc.abstractmethod
    def eval_thresh(self, img, mask=None):
        pass

    def eval(self, img, mask=None):
        self.thresh = np.empty_like(img)
        for num in xrange(len(img)):
            self.thresh[num] = self.eval_thresh(
                img[num],
                mask=mask[num] if mask is not None else None,
            )
        self.bn = (img > self.thresh)

    def show(self, panel, name):
        panel.show(self.bn, name='{}_{}_bn'.format(name, self.name), vmin=0, vmax=1)
        panel.show(self.thresh, name='{}_{}_thresh'.format(name, self.name), vmin=0, vmax=100)


class Otsu(Thresholding):
    def __init__(self):
        super(Otsu, self).__init__()
        self.name = 'otsu'

    def threshold_otsu(self, img, mask=None):
        if mask is not None:
            img = img[mask]
        return skimage.filters.threshold_otsu(img)

    def eval_thresh(self, img, mask=None):
        thresh = np.empty_like(img)
        thresh[:] = self.threshold_otsu(img, mask=mask)
        return thresh


class LocalOtsu(Otsu):
    def __init__(self, size, step):
        super(LocalOtsu, self).__init__()
        self.size = size
        self.step = step
        self.name = 'local_otsu_{size[0]}x{size[1]}_{step[0]}x{step[1]}'.format(size=size, step=step)

    def eval_thresh(self, img, mask=None):
        num_rows = (img.shape[0] - self.size[0]) / self.step[0] + 1
        num_cols = (img.shape[1] - self.size[1]) / self.step[1] + 1
        (num_rows_mat, num_cols_mat) = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing='ij')

        thresh = np.empty((num_rows, num_cols))
        for num_row in xrange(num_rows):
            for num_col in xrange(num_cols):
                begin_row = num_rows_mat[num_row, num_col] * self.step[0]
                begin_col = num_cols_mat[num_row, num_col] * self.step[1]

                img_patch = img[
                    begin_row:begin_row + self.size[0],
                    begin_col:begin_col + self.size[1],
                ]
                if mask is not None:
                    mask_patch = mask[
                        begin_row:begin_row + self.size[0],
                        begin_col:begin_col + self.size[1],
                    ]
                else:
                    mask_patch = None
                thresh[num_row, num_col] = Otsu.threshold_otsu(self, img_patch, mask=mask_patch)

        if num_rows == 1:
            num_rows = 2
            thresh = np.tile(thresh, (num_rows, 1))

        if num_cols == 1:
            num_cols = 2
            thresh = np.tile(thresh, (1, num_cols))

        interp_func = scipy.interpolate.RectBivariateSpline(
            np.arange(num_rows) * self.step[0] + self.size[0] / 2.,
            np.arange(num_cols) * self.step[1] + self.size[1] / 2.,
            thresh,
            kx=1,
            ky=1,
        )

        thresh = interp_func(
            np.arange(img.shape[0]),
            np.arange(img.shape[1]),
        )
        return thresh
