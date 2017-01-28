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

    def show(self, img, name, vmin=None, vmax=None, cmap='gray'):
        img = np.float32(img)

        plot.figure()
        img = np.reshape(img, (self.num_rows, self.num_cols, self.height, self.width, -1))
        img = np.transpose(img, (0, 2, 1, 3, 4))
        img = np.reshape(img, (self.num_rows * self.height, self.num_cols * self.width, -1))
        img = np.squeeze(img)
        plot.imsave('{}.png'.format(name), img, vmin=vmin, vmax=vmax, cmap=cmap)

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


class Process(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.name = 'processing'

    @abc.abstractmethod
    def eval(self):
        pass

    @abc.abstractmethod
    def show(self):
        pass


class Color(Process):
    def __init__(self):
        super(Color, self).__init__()
        self.name = 'coloring'

    def eval(self, img):
        self.img = skimage.color.rgb2lab(img)[..., 0]
        # self.img = np.sqrt(np.mean(np.square(skimage.color.rgb2lab(img)), axis=-1))

    def show(self, panel, name):
        panel.show(self.img, name='{}_{}_img'.format(name, self.name), vmin=0, vmax=100)


class Accumulate(Process):
    @staticmethod
    def decorate(func):
        def decorated_func(img, axis):
            img_ = np.empty_like(img)
            temp = 0
            for num in xrange(img.shape[axis]):
                slice_obj = [slice(None),] * img.ndim
                slice_obj[axis] = slice(num, num + 1)
                slice_obj = tuple(slice_obj)

                temp = func(temp, img[slice_obj])
                img_[slice_obj] = temp
            return img_
        return decorated_func

    def __init__(self):
        super(Accumulate, self).__init__()
        self.name = 'accumulating'

    def eval(self, img):
        count_fn = Accumulate.decorate(lambda x, y: (x + 1) * (y > 0))
        thresh_fn = Accumulate.decorate(lambda x, y: np.maximum(x, y) * (y > 0))

        img = np.float32(img)
        self.counts = np.empty((2,) + img.shape)
        for (count, axis) in zip(self.counts, [1, 2]):
            count_ = count_fn(img, axis)
            count_ = np.flip(thresh_fn(np.flip(count_, axis), axis), axis)
            count[:] = count_

        self.bn = np.logical_and.reduce(self.counts > 1)

    def show(self, panel, name):
        panel.show(self.counts[0], name='{}_{}_row'.format(name, self.name), vmin=0, vmax=2, cmap='jet')
        panel.show(self.counts[1], name='{}_{}_col'.format(name, self.name), vmin=0, vmax=2, cmap='jet')
        panel.show(self.bn, name='{}_{}'.format(name, self.name))


class Threshold(Process):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(Threshold, self).__init__()
        self.name = 'thresholding'

    @abc.abstractmethod
    def eval_thresh(self, img, mask):
        pass

    def eval(self, img, mask=None):
        if mask is None:
            mask = np.ones_like(img, dtype=np.bool)

        self.mask = mask
        self.thresh = np.empty_like(img)
        self.temps = np.empty((2,) + img.shape)  #
        for num in xrange(len(img)):
            self.thresh[num] = self.eval_thresh(
                img[num],
                mask=mask[num],
            )
            if hasattr(self, 'temps_'):
                for (temp, temp_) in zip(self.temps, self.temps_):
                    temp[num] = temp_
        self.bn = np.logical_and(img >= self.thresh, mask)
        self.bn_inv = np.logical_and(img < self.thresh, mask)

    def show(self, panel, name):
        panel.show(self.bn, name='{}_{}_bn'.format(name, self.name))
        panel.show(self.bn_inv, name='{}_{}_bn_inv'.format(name, self.name))
        panel.show(self.thresh, name='{}_{}_thresh'.format(name, self.name), vmin=0, vmax=100)


class Otsu(Threshold):
    def __init__(self):
        super(Otsu, self).__init__()
        self.name = 'otsu'

    def threshold_otsu(self, img, mask):
        return skimage.filters.threshold_otsu(img[mask])

    def eval_thresh(self, img, mask):
        thresh = np.full_like(img, fill_value=self.threshold_otsu(img, mask))
        return thresh


class LocalOtsu(Otsu):
    def __init__(self, size, step=None):
        super(LocalOtsu, self).__init__()
        self.size = size
        self.step = step or size
        self.name = 'local_otsu_{size[0]}x{size[1]}_{step[0]}x{step[1]}'.format(size=size, step=step)

    def eval_thresh(self, img, mask):
        num_rows = (img.shape[0] - self.size[0]) / self.step[0] + 1
        num_cols = (img.shape[1] - self.size[1]) / self.step[1] + 1

        thresh = np.zeros_like(img)
        thresh_div = np.zeros_like(img)

        img_mean = np.mean(img)  #
        temps_ = np.zeros((self.temps.shape[0],) + img.shape)

        for num_row in xrange(num_rows):
            for num_col in xrange(num_cols):
                begin_row = num_row * self.step[0]
                end_row = begin_row + self.size[0]
                begin_col = num_col * self.step[1]
                end_col = begin_col + self.size[1]

                sel = np.ix_(
                    np.arange(begin_row, end_row),
                    np.arange(begin_col, end_col),
                )

                img_patch = img[sel]
                mask_patch = mask[sel]
                # handle empty patch: currently unimplemented
                assert np.any(mask_patch), '({}, {}) mask empty!'.format(num_row, num_col)

                thresh_ = Otsu.threshold_otsu(self, img_patch, mask_patch)
                thresh[sel] += thresh_
                thresh_div[sel] += 1

                img_patch_masked = img_patch[mask_patch]
                img_patch_bn = img_patch_masked >= thresh_

                ratio = np.true_divide(np.sum(img_patch_bn), np.sum(mask_patch))
                balance = np.sqrt(ratio * (1 - ratio))
                difference = np.mean(img_patch_masked[img_patch_bn]) - np.mean(img_patch_masked[~img_patch_bn])
                average = np.mean(img_patch_masked[img_patch_bn]) + np.mean(img_patch_masked[~img_patch_bn])

                # temp[sel] += np.sqrt(ratio * (1 - ratio)) * difference

                temps_[0][sel] += (0.5 - balance)
                temps_[1][sel] += img_mean - np.mean(img_patch)

        self.temps_ = temps_ / thresh_div  #
        return thresh / thresh_div
