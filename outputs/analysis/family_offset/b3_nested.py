"""Nested (leave-one-family-out) subset selection: the held-out family influences neither the
subset choice nor the z-normalisation, so the reported numbers are honest for the whole procedure."""
import pandas as pd, numpy as np, itertools
from scipy.stats import spearmanr
S='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/'
M=pd.read_csv(S+'battery_merged.csv')
READ=[c for c in M.columns if c.startswith('mAP_')]
POOL=READ+['probe_epoch1']
rng=np.random.default_rng(0)
def protocols(s,d,tag):
    d=d.copy(); d['s']=np.asarray(s,float); d=d.dropna(subset=['s','MLLM_Avg'])
    fs=list(d.family.unique()); rs=[];rg=[]
    for _ in range(4000):
        idx=[rng.choice(d.index[d.family==f]) for f in fs]; g=d.loc[idx]
        rs.append(spearmanr(g.s,g.MLLM_Avg).statistic)
    for _ in range(8000):
        pick=rng.choice(len(fs),min(5,len(fs)),replace=False)
        idx=[rng.choice(d.index[d.family==fs[j]]) for j in pick]; g=d.loc[idx]
        rg.append(g.MLLM_Avg.max()-g.MLLM_Avg.values[g.s.values.argmax()])
    top=d[d.MLLM_Avg>=d.MLLM_Avg.quantile(.75)]
    return dict(tag=tag,n=len(d),rho=spearmanr(d.s,d.MLLM_Avg).statistic,onefam=np.mean(rs),
                regret=np.mean(rg),top25=spearmanr(top.s,top.MLLM_Avg).statistic)
def show(r): print('%-38s %5d %+7.3f %+8.3f %8.2f %+8.3f'%(r['tag'],r['n'],r['rho'],r['onefam'],r['regret'],r['top25']))

CANDS=[]
for k in (2,3,4):
    CANDS += [list(c) for c in itertools.combinations(POOL,k)]
print('candidate subsets:',len(CANDS))
def between_rho(df,cols):
    z=((df[cols]-df[cols].mean())/(df[cols].std()+1e-9)).mean(1)
    t=df.assign(z=z).groupby('family').agg(z=('z','mean'),y=('MLLM_Avg','mean'))
    return spearmanr(t.z,t.y).statistic
pred=pd.Series(np.nan,index=M.index); chosen=[]
for f in M.family.unique():
    tr=M[M.family!=f]; te=M[M.family==f]
    best=max(CANDS,key=lambda c: between_rho(tr,c))
    mu,sd=tr[best].mean(),tr[best].std()+1e-9
    pred.loc[te.index]=((te[best]-mu)/sd).mean(1).values
    chosen.append((f,tuple(best)))
print('\n%-38s %5s %7s %8s %8s %8s'%('score','n','rho','1perfam','regret','top25'))
show(protocols(pred,M,'nested subset selection (honest)'))
from collections import Counter
cc=Counter(x for _,c in chosen for x in c)
print('\n被各折选中的次数（共 %d 折）:'%len(chosen))
for k,v in cc.most_common(): print('   %-16s %d'%(k,v))

print('\n--- 预先指定的固定电池（不做任何选择，等权 z-score）---')
PRESPEC={'VTB-4  obj80+small+crowd+w_act':['mAP_obj80','mAP_small','mAP_crowd','mAP_w_act'],
         'VTB-5  +w_obj':['mAP_obj80','mAP_small','mAP_crowd','mAP_w_act','mAP_w_obj'],
         'VTB-3  obj80+small+crowd':['mAP_obj80','mAP_small','mAP_crowd'],
         'VTB-4 + probing':['mAP_obj80','mAP_small','mAP_crowd','mAP_w_act','probe_epoch1']}
for nm,c in PRESPEC.items():
    z=((M[c]-M[c].mean())/M[c].std()).mean(1); show(protocols(z,M,nm))
print('\n--- 对照 ---')
for c,sg,nm in [('probe_epoch1',1,'ImageNet probing'),('A_score',1,'A score'),
                ('mean_domain_rank',-1,'loss proxy (gated)')]:
    
    if c in M:
        sub=M.dropna(subset=[c]).reset_index(drop=True); show(protocols(sg*sub[c].values,sub,nm))
