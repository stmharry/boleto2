import bs4
import os
import requests

_ = lambda url: os.path.join('http://railway.hinet.net', url)

sess = requests.Session()
sess.headers.update({
    'Referer': _('ctno1.htm'),
})

r = sess.post(
    _('check_ctno1.jsp'),
    data={
        'person_id': 'N125584643',
        'from_station': '100',
        'to_station': '101',
        'getin_date': '2017/01/17-00',
        'train_no': '51',
        'order_qty_str': '1',
        't_order_qty_str': '0',
        'n_order_qty_str': '0',
        'd_order_qty_str': '0',
        'b_order_qty_str': '0',
        'z_order_qty_str': '0',
        'returnTicket': '0',
    },
)

with open('r.html', 'w') as f:
    f.write(r.content)

soup = bs4.BeautifulSoup(r.content, 'html.parser')
img = soup.find('img', id='idRandomPic')

r = sess.get(_(img['src'].lstrip('/')))

with open('img.jpg', 'w') as f:
    f.write(r.content)
