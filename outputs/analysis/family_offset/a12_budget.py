"""Sample-complexity of language read-out: is the family offset an ALIGNMENT-BUDGET effect?
The MLLM aligns under a fixed, small data budget. Measure R^2(caption | features) as a function
of the number of image-text pairs, not just at the endpoint."""
import numpy as np, pandas as pd, glob, os, sys
R='/cache/ma-user/VTBenchLab/lar/'; OUT='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/'
E=np.load(R+'text/caption__coco4618.npy').astype(np.float64); E-=E.mean(0)
U,S,Vt=np.linalg.svd(E,full_matrices=False); Y=(U[:,:32]*S[:32]); Y=Y/Y.std(0)
N=len(Y); rng=np.random.default_rng(0); perm=rng.permutation(N)
te=perm[3400:]; va=perm[2400:3400]; pool=perm[:2400]
sizes=[100,250,600,1500,2400]; alphas=np.logspace(-3,6,19)
def fit(X,tr):
    U,Sv,Vt=np.linalg.svd(X[tr],full_matrices=False); UtY=U.T@Y[tr]
    best=(-9,None)
    for a in alphas:
        W=Vt.T@((Sv/(Sv**2+a))[:,None]*UtY)
        r2=1-((Y[va]-X[va]@W)**2).sum()/(Y[va]**2).sum()
        if r2>best[0]: best=(r2,W)
    return 1-((Y[te]-X[te]@best[1])**2).sum()/(Y[te]**2).sum()
names=sorted([os.path.basename(f).split('__coco4618.npy')[0] for f in glob.glob(R+'features/*__coco4618.npy') if '.limit1.' not in f])
rows=[]
for k,nm in enumerate(names):
    Z=np.load(R+'features/%s__coco4618.npy'%nm).astype(np.float64); Z-=Z.mean(0); Z/= (Z.std(0)+1e-8)
    r={'name':nm}
    for n in sizes:
        vals=[fit(Z,pool[s*n:(s+1)*n]) for s in range(min(3,2400//n))]
        r['r2_n%d'%n]=float(np.mean(vals))
    r['sat_250_2400']=r['r2_n250']/max(r['r2_n2400'],1e-6)
    r['sat_600_2400']=r['r2_n600']/max(r['r2_n2400'],1e-6)
    rows.append(r); print('%3d %-32s '%(k,nm)+' '.join('n%d=%.3f'%(n,r['r2_n%d'%n]) for n in sizes)+' sat250=%.3f'%r['sat_250_2400']); sys.stdout.flush()
    pd.DataFrame(rows).to_csv(OUT+'budget.csv',index=False)
print('done')
