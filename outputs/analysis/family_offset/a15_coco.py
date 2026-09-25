import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
S='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/'
D=pd.read_csv(S+'merged.csv'); C=pd.read_csv(S+'coco_probes.csv')
M=D.merge(C,on='name').dropna(subset=['MLLM_Avg','probe_epoch1'])
print('n=%d families=%d'%(len(M),M.family.nunique()))
rng=np.random.default_rng(0)
def oneper(col,df=None):
    df=M if df is None else df; rs=[]
    fs=list(df.family.unique())
    for _ in range(5000):
        idx=[rng.choice(df.index[df.family==f]) for f in fs]; g=df.loc[idx]
        rs.append(spearmanr(g[col],g.MLLM_Avg).statistic)
    return np.mean(rs),np.std(rs)
def lofo(feats,df=None):
    df=(M if df is None else df).dropna(subset=feats); pr=[]
    for f in df.family.unique():
        tr=df[df.family!=f]; te=df[df.family==f]
        r=LinearRegression().fit(tr[feats],tr.MLLM_Avg); pr.append(pd.DataFrame({'y':te.MLLM_Avg,'p':r.predict(te[feats])}))
    P=pd.concat(pr); return spearmanr(P.p,P.y).statistic
def regretfam(col,k=5,n=8000):
    df=M.dropna(subset=[col]); fs=list(df.family.unique()); r=[]
    for _ in range(n):
        pick=rng.choice(len(fs),k,replace=False)
        idx=[rng.choice(df.index[df.family==fs[j]]) for j in pick]
        g=df.loc[idx]; r.append(g.MLLM_Avg.max()-g.MLLM_Avg.values[g[col].values.argmax()])
    return np.mean(r)
cands=['probe_epoch1','mAP_dom','mAP_obj80','mAP_small','mAP_word','mAP_word_head','mAP_word_tail','r2_raw','A_score']
print('\n%-16s %8s %8s %8s %8s'%('readout','rho_all','rho_1fam','LOFO','regret'))
for c in cands:
    if c not in M: continue
    sub=M.dropna(subset=[c])
    if len(sub)==len(M):
        m,sd=oneper(c); print('%-16s %+8.3f %+8.3f %+8.3f %8.2f  (n=%d)'%(c,spearmanr(sub[c],sub.MLLM_Avg).statistic,m,lofo([c]),regretfam(c),len(sub)))
    else:
        print('%-16s %+8.3f %8s %+8.3f %8s  (n=%d)'%(c,spearmanr(sub[c],sub.MLLM_Avg).statistic,'-',lofo([c],sub),'-',len(sub)))
print('\ncombinations (LOFO rho):')
for f in [['probe_epoch1'],['mAP_word_head'],['mAP_small'],['probe_epoch1','mAP_word_head'],
          ['probe_epoch1','mAP_small'],['mAP_word_head','mAP_small'],['probe_epoch1','r2_raw'],
          ['probe_epoch1','mAP_word_head','r2_raw'],['mAP_obj80','mAP_small','mAP_word_head'],['probe_epoch1','A_score']]:
    print('  %-45s %+.3f'%('+'.join(f),lofo(f)))
print('\ntop-25% GT subset:')
top=M[M.MLLM_Avg>=M.MLLM_Avg.quantile(0.75)]
for c in cands:
    sub=top.dropna(subset=[c]); print('  %-16s rho=%+.3f (n=%d)'%(c,spearmanr(sub[c],sub.MLLM_Avg).statistic,len(sub)))
print('\nfamily means:')
print(M.groupby('family').agg(n=('name','count'),res=('res','mean'),probe=('probe_epoch1','mean'),mllm=('MLLM_Avg','mean'),
  dom=('mAP_dom','mean'),obj80=('mAP_obj80','mean'),small=('mAP_small','mean'),
  head=('mAP_word_head','mean'),tail=('mAP_word_tail','mean')).sort_values('res').round(3).to_string())
