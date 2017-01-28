import numpy as np

from util import Panel, Color, Otsu, LocalOtsu

panel = Panel(num_rows=13, num_cols=6, height=60, width=200)

img = np.load('imgs.npy')[:panel.num] / 255.0
panel.show(img, name='rgb')

color = Color()
color.eval(img)
color.show(panel, name=str(0))

otsu = Otsu()
otsu.eval(color.img, mask=None)
otsu.show(panel, name=str(1))

from util import Accumulate
accum = Accumulate()
accum.eval(otsu.bn_inv)
accum.show(panel, name=str(2))

accum1 = Accumulate()
accum1.eval(accum.counts[1] > 1)
accum1.show(panel, name=str(3))


'''
local_otsu = LocalOtsu(size=(60, 30), step=(60, 10))
local_otsu.eval(color.img, mask=otsu.bn_inv)
local_otsu.show(panel, name=str(2))

panel.show(local_otsu.temps[0], name='temp0', vmin=0, vmax=0.5)
panel.show(local_otsu.temps[1], name='temp1', vmin=-3, vmax=3)
'''
