"""Builds the standalone site: python3 build.py  ->  index.html (one self-contained file, images embedded)."""
import base64, json, pathlib, re
R=pathlib.Path(__file__).resolve().parent; A=R/'assets'
names=['hero1','hero2','pond1','pond2','kitchen','atlas']+[f'p{i}' for i in range(6)]+[f'g{i}' for i in range(8)]
img={n:'data:image/webp;base64,'+base64.b64encode((A/f'{n}.webp').read_bytes()).decode() for n in names}
meta=json.load(open(A/'meta.json', encoding='utf-8'))
h=(R/'src/index.html').read_text(encoding='utf-8')
h=h.replace('/*PHYS*/',(R/'src/physics.js').read_text(encoding='utf-8'))
assert '/*IMG*/{}' in h and '/*META*/{}' in h
h=h.replace('/*IMG*/{}',json.dumps(img,separators=(',',':'))).replace('/*META*/{}',json.dumps(meta,separators=(',',':')))
title=re.search(r'<title>.*?</title>',h).group(0); h=h.replace(title,'',1)
page=('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
      +title+'</head><body>'+h+'</body></html>')
(R/'index.html').write_text(page, encoding='utf-8')
print('index.html',round(len(page)/1e6,2),'MB')
