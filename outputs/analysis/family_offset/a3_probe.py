import numpy as np, pandas as pd, glob, os, json, sys
R='/cache/ma-user/VTBenchLab/lar/'
OUT='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/'
rng=np.random.default_rng(0)

def load_text(dom):
    E=np.load(R+'text/%s__coco4618.npy'%dom).astype(np.float64)
    E=E-E.mean(0)
    # top-32 PCA, unit-variance per component
    U,S,Vt=np.linalg.svd(E,full_matrices=False)
    Y=(U[:,:32]*S[:32])
    Y=Y/Y.std(0)
    return Y

def ridge_r2(X,Y,tr,va,te,alphas):
    # X centered/scaled already; SVD-based ridge path
    Xt=X[tr]; Yt=Y[tr]
    U,S,Vt=np.linalg.svd(Xt,full_matrices=False)
    UtY=U.T@Yt
    best=(-9,None)
    for a in alphas:
        Dg=S/(S**2+a)
        W=Vt.T@(Dg[:,None]*UtY)
        pv=X[va]@W
        r2=1-((Y[va]-pv)**2).sum()/ (Y[va]**2).sum()
        if r2>best[0]: best=(r2,W)
    W=best[1]
    pt=X[te]@W
    return 1-((Y[te]-pt)**2).sum()/(Y[te]**2).sum()

def krr_r2(X,Y,tr,va,te,gammas,alphas):
    def K(A,B,g): 
        d2=(A**2).sum(1)[:,None]+(B**2).sum(1)[None,:]-2*A@B.T
        return np.exp(-g*np.maximum(d2,0))
    best=(-9,None,None)
    Ktr=None
    for g in gammas:
        Ktt=K(X[tr],X[tr],g); Kv=K(X[va],X[tr],g)
        n=len(tr)
        w,V=np.linalg.eigh(Ktt)
        VtY=V.T@Y[tr]
        for a in alphas:
            A=V@(VtY/(w+a)[:,None])
            r2=1-((Y[va]-Kv@A)**2).sum()/(Y[va]**2).sum()
            if r2>best[0]: best=(r2,g,a)
    _,g,a=best
    Ktt=K(X[tr],X[tr],g); w,V=np.linalg.eigh(Ktt); A=V@((V.T@Y[tr])/(w+a)[:,None])
    Kte=K(X[te],X[tr],g)
    return 1-((Y[te]-Kte@A)**2).sum()/(Y[te]**2).sum()

def mutual_knn(Z,E,k=10):
    def nn(A):
        A=A-A.mean(0); A=A/ (np.linalg.norm(A,axis=1,keepdims=True)+1e-9)
        S=A@A.T; np.fill_diagonal(S,-9)
        return np.argpartition(-S,k,axis=1)[:,:k]
    a=nn(Z); b=nn(E)
    return np.mean([len(set(a[i])&set(b[i]))/k for i in range(len(a))])

def cka(Z,E):
    Z=Z-Z.mean(0); E=E-E.mean(0)
    zz=Z.T@Z; ee=E.T@E; ze=Z.T@E
    return (ze**2).sum()/np.sqrt((zz**2).sum()*(ee**2).sum())

Ycap=load_text('caption')
N=Ycap.shape[0]
perm=rng.permutation(N)
tr,va,te=perm[:2400],perm[2400:3400],perm[3400:]
alphas=np.logspace(-3,5,17); gammas=None

names=sorted([os.path.basename(f).split('__coco4618.npy')[0] for f in glob.glob(R+'features/*__coco4618.npy') if '.limit1.' not in f])
print('n encoders',len(names)); sys.stdout.flush()
rows=[]
for i,nm in enumerate(names):
    Z=np.load(R+'features/%s__coco4618.npy'%nm).astype(np.float64)
    meta=json.load(open(R+'features/%s__coco4618.meta.json'%nm))
    Zc=Z-Z.mean(0)
    sd=Zc.std(0)+1e-8
    Zs=Zc/sd
    # spectral
    lam=np.linalg.svd(Zc,compute_uv=False)**2/(N-1)
    p=lam/lam.sum()
    top1=p[0]; pr=(lam.sum()**2)/(lam**2).sum()
    er=np.exp(-(p*np.log(p+1e-30)).sum())
    # PCA-128 for KRR (keeps cost fixed across d)
    U,S,Vt=np.linalg.svd(Zc,full_matrices=False)
    P=(U[:,:128]*S[:128]); P=P/P.std(0)   # whitened top-128
    med=np.median(((P[:500][:,None,:]-P[:500][None,:,:])**2).sum(-1))
    gs=[g/med for g in [0.5,1.0,2.0,4.0]]
    r2_raw=ridge_r2(Zc/np.linalg.norm(Zc,axis=0).mean(),Ycap,tr,va,te,alphas)
    r2_std=ridge_r2(Zs,Ycap,tr,va,te,alphas)
    r2_krr=krr_r2(P,Ycap,tr,va,te,gs,np.logspace(-4,2,13))
    r2_lin_p=ridge_r2(P,Ycap,tr,va,te,alphas)
    rows.append(dict(name=nm,d=meta['d'],n_tokens=meta['n_tokens'],top1_share=top1,part_ratio=pr,eff_rank_e=er,
        r2_raw=r2_raw,r2_std=r2_std,r2_krr=r2_krr,r2_lin_p=r2_lin_p,
        gap_scale=r2_std-r2_raw,gap_nonlin=r2_krr-r2_lin_p,
        mknn=mutual_knn(Zc,Ycap),cka_txt=cka(Zc,Ycap)))
    print('%3d %-32s d=%4d tok=%4d  r2raw=%.3f r2std=%.3f r2lin_p=%.3f r2krr=%.3f mknn=%.3f'%(i,nm,meta['d'],meta['n_tokens'],r2_raw,r2_std,r2_lin_p,r2_krr,rows[-1]['mknn'])); sys.stdout.flush()
pd.DataFrame(rows).to_csv(OUT+'struct.csv',index=False)
print('saved')
