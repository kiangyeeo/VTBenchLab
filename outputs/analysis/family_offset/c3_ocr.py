"""Text-reading read-outs on TextVQA images. TextVQA is 24-26% of the gold label's variance and
had no probe at all. oi_class is the control: same images, same probe, object labels instead of
text labels - it separates "OCR ability" from "just a different image domain"."""
import numpy as np, pandas as pd, json, glob, os, re, sys
T='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/textvqa/'
OUT='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/'
ids=[l.strip() for l in open(T+'ids.txt') if l.strip()]; N=len(ids)
lab=json.load(open(T+'labels.json'))
from collections import Counter
def build(get,minc,capn):
    c=Counter()
    for i in ids: c.update(get(lab[i]))
    V=[w for w,n in c.most_common() if n>=minc][:capn]
    vm={w:j for j,w in enumerate(V)}; Y=np.zeros((N,len(V)),np.float32)
    for k,i in enumerate(ids):
        for w in get(lab[i]):
            if w in vm: Y[k,vm[w]]=1
    return Y,V
Yword,Vw=build(lambda r:{w.upper() for w in r['ocr'] if re.match(r'^[A-Za-z]{2,}$',w)},8,400)
Ynum,Vn =build(lambda r:{w for w in r['ocr'] if re.match(r'^[0-9]{1,4}$',w)},6,120)
Yoi,Voi =build(lambda r:set(r['classes']),8,400)
ntok=np.array([len([w for w in lab[i]['ocr']]) for i in ids],float)
Ydense=np.stack([(ntok>=q).astype(np.float32) for q in [3,6,10,18,30]],1)
print('ocr_word=%d  ocr_num=%d  oi_class=%d  dense=%d  (N=%d)'%(len(Vw),len(Vn),len(Voi),Ydense.shape[1],N))
print('pos/img: word=%.2f num=%.2f oi=%.2f'%(Yword.sum(1).mean(),Ynum.sum(1).mean(),Yoi.sum(1).mean()))
TASKS=[('ocr_word',Yword),('ocr_num',Ynum),('ocr_dense',Ydense),('oi_class',Yoi)]

rng=np.random.default_rng(0); perm=rng.permutation(N)
n1,n2=int(N*.55),int(N*.75)
tr,va,te=perm[:n1],perm[n1:n2],perm[n2:]
alphas=np.logspace(-1,5,10)
def ap(Yt,P):
    o=[]
    for j in range(Yt.shape[1]):
        y=Yt[:,j]
        if y.sum()<3: continue
        s=np.argsort(-P[:,j]); ys=y[s]; tp=np.cumsum(ys)
        o.append(((tp/np.arange(1,len(ys)+1))*ys).sum()/y.sum())
    return float(np.mean(o)) if o else np.nan
rows=[]
for k,f in enumerate(sorted(glob.glob(T+'feat/*.npy'))):
    nm=os.path.basename(f)[:-4]
    Z=np.load(f).astype(np.float64); Z=Z-Z.mean(0); Z=Z/(Z.std(0)+1e-8)
    U,S,Vt=np.linalg.svd(Z[tr],full_matrices=False)
    r={'name':nm}
    for tn,Y in TASKS:
        UtY=U.T@Y[tr]; best=(-9,None)
        for a in alphas:
            W=Vt.T@((S/(S**2+a))[:,None]*UtY); m=ap(Y[va],Z[va]@W)
            if m>best[0]: best=(m,W)
        r['mAP_'+tn]=ap(Y[te],Z[te]@best[1])
    rows.append(r); print('%3d %-32s '%(k,nm)+' '.join('%s=%.3f'%(t,r['mAP_'+t]) for t,_ in TASKS)); sys.stdout.flush()
    pd.DataFrame(rows).to_csv(OUT+'ocr_probes.csv',index=False)
print('saved')
