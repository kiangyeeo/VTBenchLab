"""Readout battery v2: one SVD per encoder, many label spaces.
Adds counting, and POS-partitioned caption vocabulary (objects / attributes / relations / actions),
which target the VQA/GQA/caption capabilities that ImageNet 1000-way single-label cannot express."""
import numpy as np, pandas as pd, json, glob, os, sys, re
from collections import Counter
R='/cache/ma-user/VTBenchLab/'
OUT='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/'
ids=[l.strip() for l in open(R+'lar/features/dinov2_large__coco4618.ids.txt')]
iid=[int(x) for x in ids]; pos={v:i for i,v in enumerate(iid)}; N=len(ids)

inst=json.load(open(R+'data/gvt/raw/coco/annotations/instances_val2017.json'))
cats=sorted({c['id'] for c in inst['categories']}); cmap={c:i for i,c in enumerate(cats)}
K=len(cats)
Ypres=np.zeros((N,K),np.float32); Cnt=np.zeros((N,K),np.float32); Area=np.zeros((N,K),np.float32)
Small=np.zeros((N,K),np.float32)
for a in inst['annotations']:
    if a['image_id'] in pos:
        i,j=pos[a['image_id']],cmap[a['category_id']]
        Ypres[i,j]=1; Cnt[i,j]+=1; Area[i,j]+=a['area']
        if a['area']<4000: Small[i,j]=1
Ydom=np.zeros_like(Ypres)
for i in range(N):
    if Area[i].max()>0: Ydom[i,Area[i].argmax()]=1
Ycount2=(Cnt>=2).astype(np.float32)          # "at least two of category c"
Ycount3=(Cnt>=4).astype(np.float32)          # crowded scenes

cap=json.load(open(R+'data/gvt/raw/coco/annotations/captions_val2017.json'))
byimg={}
for a in cap['annotations']:
    if a['image_id'] in pos: byimg.setdefault(pos[a['image_id']],[]).append(a['caption'].lower())
COLOR=set('red blue green white black yellow brown orange pink gray grey purple silver golden dark bright colorful striped'.split())
REL=set('on in at near next behind under above below beside front back top bottom left right between across around beneath inside outside onto over through against along beside'.split())
NUM=set('one two three four five six several many few group pair couple bunch multiple crowd row line stack pile'.split())
ACT=set('sitting standing holding riding playing eating walking running looking flying jumping laying lying hanging watching talking wearing carrying throwing catching cutting driving swimming skiing surfing'.split())
STOP=set('a an the of and or with is are was were be been for from by this that there here it its his her their they he she we you as into very some no not while during after before'.split())
toks={}; cnt=Counter()
for i,cs in byimg.items():
    w=set()
    for c in cs: w|={t for t in re.findall(r"[a-z]+",c) if len(t)>2 and t not in STOP}
    toks[i]=w; cnt.update(w)
def make(vocab,minc=40,cap_n=300):
    V=[w for w,c in cnt.most_common(4000) if c>=minc and w in vocab][:cap_n]
    if not V: return None,[]
    vm={w:j for j,w in enumerate(V)}; Y=np.zeros((N,len(V)),np.float32)
    for i,w in toks.items():
        for t in w:
            if t in vm: Y[i,vm[t]]=1
    return Y,V
OBJV=set(w for w,c in cnt.most_common(1200) if c>=40) - COLOR - REL - NUM - ACT
Yobjw,_=make(OBJV,40,300); Ycol,vc=make(COLOR,25); Yrel,vr=make(REL,25); Ynum,vn=make(NUM,25); Yact,va=make(ACT,25)
print('vocab sizes: obj=%d color=%d rel=%d num=%d act=%d'%(Yobjw.shape[1],len(vc),len(vr),len(vn),len(va)))

TASKS=[('dom',Ydom),('obj80',Ypres),('small',Small),('count2',Ycount2),('crowd',Ycount3),
       ('w_obj',Yobjw),('w_color',Ycol),('w_rel',Yrel),('w_num',Ynum),('w_act',Yact)]
for n,Y in TASKS: print('  %-8s classes=%3d  pos/img=%.2f'%(n,Y.shape[1],Y.sum(1).mean()))

rng=np.random.default_rng(0); perm=rng.permutation(N)
tr,va_,te=perm[:2400],perm[2400:3400],perm[3400:]
alphas=np.logspace(-1,5,10)

def ap_matrix(Yt,P):
    """vectorised average precision per column"""
    out=[]
    for j in range(Yt.shape[1]):
        y=Yt[:,j]; n1=y.sum()
        if n1<3: continue
        o=np.argsort(-P[:,j]); ys=y[o]
        tp=np.cumsum(ys); prec=tp/np.arange(1,len(ys)+1)
        out.append((prec*ys).sum()/n1)
    return float(np.mean(out)) if out else np.nan

names=sorted([os.path.basename(f).split('__coco4618.npy')[0] for f in glob.glob(R+'lar/features/*__coco4618.npy') if '.limit1.' not in f])
print('encoders',len(names)); sys.stdout.flush()
rows=[]
for k,nm in enumerate(names):
    Z=np.load(R+'lar/features/%s__coco4618.npy'%nm).astype(np.float64)
    Z=Z-Z.mean(0); Z=Z/(Z.std(0)+1e-8)
    U,S,Vt=np.linalg.svd(Z[tr],full_matrices=False)      # ONE svd, reused by every label space
    Zv,Zt=Z[va_],Z[te]
    r={'name':nm}
    for tn,Y in TASKS:
        UtY=U.T@Y[tr]; best=(-9,None)
        for a in alphas:
            W=Vt.T@((S/(S**2+a))[:,None]*UtY)
            m=ap_matrix(Y[va_],Zv@W)
            if m>best[0]: best=(m,W)
        r['mAP_'+tn]=ap_matrix(Y[te],Zt@best[1])
    rows.append(r)
    print('%3d %-32s '%(k,nm)+' '.join('%s=%.3f'%(t,r['mAP_'+t]) for t,_ in TASKS)); sys.stdout.flush()
    pd.DataFrame(rows).to_csv(OUT+'battery.csv',index=False)
print('saved')
