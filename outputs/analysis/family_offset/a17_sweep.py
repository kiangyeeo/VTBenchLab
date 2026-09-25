import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
S='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/'
E=pd.read_csv('/cache/ma-user/VTBenchLab/gradient_compatibility/artifacts/full_sweep_v1/summary/evaluation.csv')
D=pd.read_csv(S+'merged.csv')
E['name']=E.registry_name
M=E.merge(D[['name','family','probe_epoch1','MLLM_Avg','res','A_score']],on='name',how='left',suffixes=('','_d'))
C=pd.read_csv(S+'coco_probes.csv'); M=M.merge(C,on='name',how='left')
M=M.dropna(subset=['qwen2_5'])
print('n=%d  families=%d'%(len(M),M.family.nunique()))

print('\n=== [A] 健全性：real - zero >= 0 的 run（视觉 token 比不给图还差） ===')
bad=M[(M.caption_real_minus_zero>=0)|(M.reasoning_real_minus_zero>=0)]
print('caption 或 reasoning 任一 >= 0 的：%d / %d'%(len(bad),len(M)))
hard=M[(M.caption_real_minus_zero>=0)]
print('caption >= 0（重度退化）：%d  -> %s'%(len(hard),list(hard.name)))
print('reasoning >= 0：%d'%int((M.reasoning_real_minus_zero>=0).sum()))
print('\nreasoning_real_minus_zero 分布: min=%.3f  中位=%.3f  max=%.3f'%(M.reasoning_real_minus_zero.min(),M.reasoning_real_minus_zero.median(),M.reasoning_real_minus_zero.max()))
print('caption_real_minus_zero  分布: min=%.3f  中位=%.3f  max=%.3f'%(M.caption_real_minus_zero.min(),M.caption_real_minus_zero.median(),M.caption_real_minus_zero.max()))

rng=np.random.default_rng(0)
def prot(col,sign,df=None,label=None):
    df=(M if df is None else df).dropna(subset=[col,'qwen2_5'])
    v=sign*df[col].values; y=df.qwen2_5.values
    rho=spearmanr(v,y).statistic
    fs=list(df.family.unique()); rs=[]
    for _ in range(4000):
        idx=[rng.choice(df.index[df.family==f]) for f in fs]; g=df.loc[idx]
        rs.append(spearmanr(sign*g[col],g.qwen2_5).statistic)
    r=[]
    for _ in range(8000):
        pick=rng.choice(len(fs),min(5,len(fs)),replace=False)
        idx=[rng.choice(df.index[df.family==fs[j]]) for j in pick]; g=df.loc[idx]
        r.append(g.qwen2_5.max()-g.qwen2_5.values[(sign*g[col]).values.argmax()])
    return rho,np.mean(rs),np.mean(r),len(df)

print('\n=== [B] 全池评测（目标 qwen2.5 MLLM Avg） ===')
print('%-34s %8s %10s %8s %5s'%('score','全表 rho','一族一个','跨族regret','n'))
rows=[('mean_domain_rank (他们的最终分)','mean_domain_rank',-1),
      ('caption_loss','caption_loss',-1),
      ('reasoning_loss','reasoning_loss',-1),
      ('caption_real_minus_zero','caption_real_minus_zero',-1),
      ('reasoning_real_minus_zero','reasoning_real_minus_zero',-1),
      ('caption_real_minus_shuffled','caption_real_minus_shuffled',-1),
      ('ImageNet probing (对照)','probe_epoch1',1),
      ('A_score (对照)','A_score',1),
      ('COCO obj80 mAP (对照)','mAP_obj80',1)]
for lab,c,sg in rows:
    if c not in M: continue
    rho,op,rg,n=prot(c,sg)
    print('%-34s %+8.3f %+10.3f %8.2f %5d'%(lab,rho,op,rg,n))

print('\n=== [C] 剔掉 3 个退化 run 后 ===')
clean=M[M.caption_real_minus_zero<0]
for lab,c,sg in rows[:6]:
    if c not in M: continue
    rho,op,rg,n=prot(c,sg,clean)
    print('%-34s %+8.3f %+10.3f %8.2f %5d'%(lab,rho,op,rg,n))

print('\n=== [D] 最差的 12 个预测误差 ===')
M2=M.assign(err=M.mllm_position-M.predicted_position)
print(M2.reindex(M2.err.abs().sort_values(ascending=False).index)[['name','family','predicted_position','mllm_position','err','caption_real_minus_zero','reasoning_real_minus_zero','qwen2_5']].head(12).round(3).to_string(index=False))

print('\n=== [E] 家族层面：预测名次 vs 真实名次 ===')
g=M.groupby('family').agg(n=('name','count'),pred=('predicted_position','mean'),true=('mllm_position','mean'),
   cap0=('caption_real_minus_zero','mean'),res0=('reasoning_real_minus_zero','mean')).sort_values('true')
print(g.round(2).to_string())

print('\n=== [F] 组合：probing + caption 损失 (leave-one-family-out) ===')
def lofo(feats,df=None):
    df=(M if df is None else df).dropna(subset=feats+['qwen2_5']); pr=[]
    for f in df.family.unique():
        tr=df[df.family!=f]; te=df[df.family==f]
        if len(tr)<5: continue
        r=LinearRegression().fit(tr[feats],tr.qwen2_5); pr.append(pd.DataFrame({'y':te.qwen2_5,'p':r.predict(te[feats])}))
    P=pd.concat(pr); return spearmanr(P.p,P.y).statistic,len(df)
for f in [['caption_loss'],['caption_real_minus_zero'],['mean_domain_rank'],['probe_epoch1'],
          ['probe_epoch1','caption_loss'],['probe_epoch1','caption_real_minus_zero'],
          ['mAP_obj80','caption_loss'],['probe_epoch1','A_score']]:
    if not all(c in M for c in f): continue
    r,n=lofo(f); print('  %-45s %+.3f (n=%d)'%('+'.join(f),r,n))
