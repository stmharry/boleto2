import numpy as np

from util import Panel, Coloring, Otsu, LocalOtsu

panel = Panel(num_rows=13, num_cols=6, height=60, width=200)

img = np.load('imgs.npy')[:panel.num] / 255.0
panel.show(img, name='rgb')

coloring = Coloring()
coloring.eval(img)
coloring.show(panel, name=str(0))

otsu = Otsu()
otsu.eval(coloring.img, mask=None)
otsu.show(panel, name=str(1))

local_otsu = LocalOtsu(size=(60, 30), step=(60, 15))
local_otsu.eval(coloring.img, mask=~otsu.bn)
local_otsu.show(panel, name=str(2))
