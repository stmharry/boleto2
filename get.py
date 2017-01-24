import cStringIO
import PIL.Image
import requests
import numpy as np

sess = requests.Session()

imgs = np.empty((1024, 60, 200, 3), dtype=np.uint8)
for (num, img) in enumerate(imgs):
    print(num)
    r = sess.get('http://railway.hinet.net/ImageOut.jsp')
    img[:] = np.array(PIL.Image.open(cStringIO.StringIO(r.content)))

np.save('imgs.npy', imgs)
