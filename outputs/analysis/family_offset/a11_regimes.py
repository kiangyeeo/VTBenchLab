import pandas as pd, numpy as np
from scipy.stats import spearmanr
S='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/'
D=pd.read_csv(S+'merged.csv')
print('=== [K] two decision regimes: pick ACROSS families vs pick WITHIN a family ===')
print('%-14s | %-22s | %-22s'%('','ACROSS (one-per-family)','WITHIN (family-centred)'))
rng=np.random.default_rng(0)
for m,s in [('probe_epoch1',1),('A_score',1),('r2_raw',1),('mAP',1)]:
    if m not in D: continue
    sub=D.dropna(subset=[m,'MLLM_Avg']).copy()
    fams=[f for f in sub.family.unique()]
    rs=[]
    for _ in range(4000):
        idx=[rng.choice(sub.index[sub.family==f]) for f in fams]; g=sub.loc[idx]
        rs.append(spearmanr(g[m],g.MLLM_Avg).statistic)
    sub['xc']=sub[m]-sub.groupby('family')[m].transform('mean')
    sub['yc']=sub.MLLM_Avg-sub.groupby('family').MLLM_Avg.transform('mean')
    big=sub[sub.family.map(sub.family.value_counts())>=4]
    print('%-14s | %+.3f (n_fam=%2d)        | %+.3f (n=%d)'%(m,np.mean(rs),len(fams),spearmanr(big.xc,big.yc).statistic,len(big)))

print('\n  per-family head-to-head (n>=4):')
print('  %-12s %3s %9s %9s'%('family','n','probe','A_score'))
for f,g in D.groupby('family'):
    if len(g)>=4:
        a=spearmanr(g.probe_epoch1,g.MLLM_Avg).statistic
        gg=g.dropna(subset=['A_score'])
        b=spearmanr(gg.A_score,gg.MLLM_Avg).statistic if len(gg)>=4 else float('nan')
        print('  %-12s %3d %+9.3f %+9.3f'%(f,len(g),a,b))

print('\n=== [L] the practically relevant regime: only strong candidates ===')
for q in [0.0,0.4,0.6,0.75]:
    thr=D.MLLM_Avg.quantile(q); sub=D[D.MLLM_Avg>=thr]
    sa=sub.dropna(subset=['A_score'])
    print('  top %2d%% by GT (n=%2d): probe rho=%+.3f | A_score rho=%+.3f (n=%d)'%(
        int(100*(1-q)),len(sub),spearmanr(sub.probe_epoch1,sub.MLLM_Avg).statistic,
        spearmanr(sa.A_score,sa.MLLM_Avg).statistic,len(sa)))

print('\n=== [M] top-1 regret (pick best of k random tokenizers), k=5 and k=10 ===')
def regret(col,k,n=20000,pool=None):
    sub=(pool if pool is not None else D).dropna(subset=[col,'MLLM_Avg'])
    v=sub[col].values; y=sub.MLLM_Avg.values; r=[]
    for _ in range(n):
        i=rng.choice(len(v),k,replace=False)
        r.append(y[i].max()-y[i][v[i].argmax()])
    return np.mean(r)
for col in ['probe_epoch1','A_score','r2_raw','mknn','RankMe']:
    if col in D: print('  %-14s k=5 regret=%.2f  k=10 regret=%.2f  (n=%d)'%(col,regret(col,5),regret(col,10),D[col].notna().sum()))
# family-stratified regret: one candidate per family (the hard case)
print('\n  regret when the 5 candidates come from 5 DIFFERENT families:')
for col in ['probe_epoch1','A_score','r2_raw']:
    sub=D.dropna(subset=[col,'MLLM_Avg']); fams=list(sub.family.unique()); r=[]
    for _ in range(20000):
        fs=rng.choice(len(fams),5,replace=False)
        idx=[rng.choice(sub.index[sub.family==fams[f]]) for f in fs]
        g=sub.loc[idx]; r.append(g.MLLM_Avg.max()-g.MLLM_Avg.values[g[col].values.argmax()])
    print('  %-14s %.2f'%(col,np.mean(r)))
