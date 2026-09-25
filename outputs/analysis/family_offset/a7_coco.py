"""Does the probing failure come from the DATASET (ImageNet vs COCO), the LABEL GRANULARITY
(1000-way single label vs 80-way multi-label scene), or the VOCABULARY (class index vs caption words)?
Same cached pooled features, three different linear read-outs."""
import numpy as np, pandas as pd, json, glob, os, sys, re
from sklearn.metrics import average_precision_score
R='/cache/ma-user/VTBenchLab/'; OUT='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/'
ids=[l.strip() for l in open(R+'lar/features/dinov2_large__coco4618.ids.txt')]
iid=[int(x) for x in ids]; pos={v:i for i,v in enumerate(iid)}; N=len(ids)
print('N',N)
# --- labels A: COCO 80-way multi-label (objects present) ---
inst=json.load(open(R+'data/gvt/raw/coco/annotations/instances_val2017.json'))
cats=sorted({c['id'] for c in inst['categories']}); cmap={c:i for i,c in enumerate(cats)}
Yobj=np.zeros((N,len(cats)),dtype=np.float32)
Aarea=np.zeros((N,len(cats)),dtype=np.float32)
for a in inst['annotations']:
    if a['image_id'] in pos:
        Yobj[pos[a['image_id']],cmap[a['category_id']]]=1
        Aarea[pos[a['image_id']],cmap[a['category_id']]]+=a['area']
# labels A2: only the LARGEST object per image (ImageNet-like single dominant object)
Ydom=np.zeros_like(Yobj)
for i in range(N):
    if Aarea[i].max()>0: Ydom[i,Aarea[i].argmax()]=1
# labels A3: SMALL objects only (area < 2% of a 640x480-ish image) -> fine detail
Ysmall=((Aarea>0)&(Aarea<4000)).astype(np.float32)
# --- labels B: caption content words ---
cap=json.load(open(R+'data/gvt/raw/coco/annotations/captions_val2017.json'))
stop=set('a an the of on in at to and or with is are was were be been for from by this that there here it its his her their they he she we you i as into over under near next up down out off very some many few no not while during after before around through'.split())
byimg={}
for a in cap['annotations']:
    if a['image_id'] in pos: byimg.setdefault(pos[a['image_id']],[]).append(a['caption'].lower())
from collections import Counter
cnt=Counter()
toks={}
for i,cs in byimg.items():
    w=set()
    for c in cs: w|= {t for t in re.findall(r"[a-z]+",c) if len(t)>2 and t not in stop}
    toks[i]=w; cnt.update(w)
vocab=[w for w,c in cnt.most_common(600) if c>=40][:500]
vmap={w:j for j,w in enumerate(vocab)}
Yword=np.zeros((N,len(vocab)),dtype=np.float32)
for i,w in toks.items():
    for t in w:
        if t in vmap: Yword[i,vmap[t]]=1
# split rare words into frequent(head) vs rare(tail) halves
freq=Yword.mean(0); order=np.argsort(-freq)
head=order[:100]; tail=order[250:500]
print('label sets: obj80 pos/img=%.2f  dom=%.2f  small=%.2f  words%d pos/img=%.2f'%(Yobj.sum(1).mean(),Ydom.sum(1).mean(),Ysmall.sum(1).mean(),len(vocab),Yword.sum(1).mean()))

rng=np.random.default_rng(0); perm=rng.permutation(N)
tr,va,te=perm[:2400],perm[2400:3400],perm[3400:]
alphas=np.logspace(-2,6,17)
def probe(X,Y,cols=None):
    U,S,Vt=np.linalg.svd(X[tr],full_matrices=False); UtY=U.T@Y[tr]
    best=(-9,None)
    for a in alphas:
        W=Vt.T@((S/(S**2+a))[:,None]*UtY)
        p=X[va]@W
        m=np.nanmean([average_precision_score(Y[va][:,j],p[:,j]) for j in range(Y.shape[1]) if Y[va][:,j].sum()>=3])
        if m>best[0]: best=(m,W)
    p=X[te]@best[1]
    idxs=range(Y.shape[1]) if cols is None else cols
    return float(np.nanmean([average_precision_score(Y[te][:,j],p[:,j]) for j in idxs if Y[te][:,j].sum()>=3]))

names=sorted([os.path.basename(f).split('__coco4618.npy')[0] for f in glob.glob(R+'lar/features/*__coco4618.npy') if '.limit1.' not in f])
rows=[]
for k,nm in enumerate(names):
    Z=np.load(R+'lar/features/%s__coco4618.npy'%nm).astype(np.float64)
    Z=Z-Z.mean(0); Z=Z/(Z.std(0)+1e-8)
    r=dict(name=nm,
        mAP_obj80=probe(Z,Yobj), mAP_dom=probe(Z,Ydom), mAP_small=probe(Z,Ysmall),
        mAP_word=probe(Z,Yword), mAP_word_head=probe(Z,Yword,head), mAP_word_tail=probe(Z,Yword,tail))
    rows.append(r); print('%3d %-32s obj80=%.3f dom=%.3f small=%.3f word=%.3f head=%.3f tail=%.3f'%(k,nm,r['mAP_obj80'],r['mAP_dom'],r['mAP_small'],r['mAP_word'],r['mAP_word_head'],r['mAP_word_tail'])); sys.stdout.flush()
pd.DataFrame(rows).to_csv(OUT+'coco_probes.csv',index=False)
print('saved')
