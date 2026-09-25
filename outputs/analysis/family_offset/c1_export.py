import pyarrow.parquet as pq, glob, io, json, os, re
from collections import Counter
from PIL import Image
OUT='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/textvqa/'
seen={}; recs={}
for f in sorted(glob.glob('/cache/ma-user/VTBenchLab/data/gradient_compatibility/textvqa/data/*.parquet')):
    t=pq.read_table(f,columns=['image_id','image','ocr_tokens','answers','image_classes'])
    for r in t.to_pylist():
        iid=r['image_id']
        if iid in recs:
            recs[iid]['answers'] += list(r['answers'] or [])
            continue
        recs[iid]={'ocr':[str(x) for x in (r['ocr_tokens'] or [])],
                   'answers':[str(x) for x in (r['answers'] or [])],
                   'classes':[str(x) for x in (r['image_classes'] or [])]}
        seen[iid]=r['image']['bytes']
print('unique images',len(recs))
os.makedirs(OUT+'img',exist_ok=True)
ids=sorted(recs)
for iid in ids:
    p=OUT+'img/%s.jpg'%iid
    if not os.path.exists(p):
        Image.open(io.BytesIO(seen[iid])).convert('RGB').save(p,quality=92)
json.dump({k:recs[k] for k in ids},open(OUT+'labels.json','w'))
open(OUT+'ids.txt','w').write('\n'.join(ids))
c=Counter()
for k in ids: c.update({w.upper() for w in recs[k]['ocr'] if re.match(r'^[A-Za-z0-9]{2,}$',w)})
print('ocr vocab: >=20 imgs:',sum(1 for w,n in c.items() if n>=20),' >=10:',sum(1 for w,n in c.items() if n>=10),' >=5:',sum(1 for w,n in c.items() if n>=5))
print('top30:',c.most_common(30))
import numpy as np
print('median #ocr tokens/img', np.median([len(recs[k]['ocr']) for k in ids]))
ca=Counter()
for k in ids: ca.update({a.lower() for a in recs[k]['answers']})
print('answer vocab >=15:',sum(1 for w,n in ca.items() if n>=15),'top20:',ca.most_common(20))
