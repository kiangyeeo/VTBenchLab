import pandas as pd, numpy as np
from scipy.stats import spearmanr, pearsonr
S='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/'
D=pd.read_csv(S+'master.csv'); T=pd.read_csv(S+'struct.csv')
M=D.merge(T,on='name',how='inner',suffixes=('','_s')).dropna(subset=['MLLM_Avg','probe_epoch1'])
print('n=',len(M))
cols=['r2_raw','r2_std','r2_lin_p','r2_krr','gap_scale','gap_nonlin','mknn','cka_txt','top1_share','part_ratio','eff_rank_e','n_tokens_s','d_s','A_score']
print('\n=== correlations with MLLM_Avg and with probing RESIDUAL ===')
print('%-12s %8s %8s %8s %8s'%('feature','rho_MLLM','rho_RES','r_RES','rho_1fam'))
rng=np.random.default_rng(0); fams=list(M.family.unique())
for c in cols:
    sub=M.dropna(subset=[c])
    rs=[]
    for _ in range(3000):
        idx=[rng.choice(sub.index[sub.family==f]) for f in fams if (sub.family==f).any()]
        s=sub.loc[idx]; rs.append(spearmanr(s[c],s.MLLM_Avg).statistic)
    print('%-12s %+8.3f %+8.3f %+8.3f %+8.3f'%(c,spearmanr(sub[c],sub.MLLM_Avg).statistic,
        spearmanr(sub[c],sub.res).statistic,pearsonr(sub[c],sub.res).statistic,np.mean(rs)))

print('\n=== family means of the key structural features ===')
g=M.groupby('family').agg(n=('name','count'),res=('res','mean'),probe=('probe_epoch1','mean'),mllm=('MLLM_Avg','mean'),
   r2raw=('r2_raw','mean'),r2std=('r2_std','mean'),r2krr=('r2_krr','mean'),gscale=('gap_scale','mean'),
   gnl=('gap_nonlin','mean'),mknn=('mknn','mean'),top1=('top1_share','mean'),tok=('n_tokens','mean')).sort_values('res')
print(g.round(3).to_string())

print('\n=== combine probing + one structural feature (linear reg), holdout by FAMILY ===')
from sklearn.linear_model import LinearRegression
def loo_family(feats):
    pr=[]
    for f in M.family.unique():
        tr=M[M.family!=f]; te=M[M.family==f]
        r=LinearRegression().fit(tr[feats],tr.MLLM_Avg)
        pr.append(pd.DataFrame({'name':te.name,'y':te.MLLM_Avg,'p':r.predict(te[feats])}))
    P=pd.concat(pr)
    return spearmanr(P.p,P.y).statistic, np.sqrt(((P.p-P.y)**2).mean())
for feats in [['probe_epoch1'],['A_score'],['r2_krr'],['mknn'],['probe_epoch1','r2_krr'],['probe_epoch1','mknn'],
              ['probe_epoch1','gap_nonlin'],['probe_epoch1','r2_std'],['probe_epoch1','A_score'],
              ['probe_epoch1','mknn','r2_krr'],['probe_epoch1','mknn','n_tokens_s']]:
    sub=M.dropna(subset=feats)
    if len(sub)<40: 
        print('%-40s SKIP n=%d'%('+'.join(feats),len(sub))); continue
    globals()['M_']=M
    pr=[]
    for f in sub.family.unique():
        tr=sub[sub.family!=f]; te=sub[sub.family==f]
        r=LinearRegression().fit(tr[feats],tr.MLLM_Avg); pr.append(pd.DataFrame({'y':te.MLLM_Avg,'p':r.predict(te[feats])}))
    P=pd.concat(pr)
    print('%-40s n=%2d  leave-one-FAMILY-out: rho=%+.3f  RMSE=%.2f'%('+'.join(feats),len(sub),spearmanr(P.p,P.y).statistic,np.sqrt(((P.p-P.y)**2).mean())))
M.to_csv(S+'merged.csv',index=False)
