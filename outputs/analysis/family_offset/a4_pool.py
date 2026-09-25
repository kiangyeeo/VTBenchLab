import pandas as pd, numpy as np
from scipy.stats import spearmanr, pearsonr
D=pd.read_csv('/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/master.csv')
mets=['probe_epoch1','retrieval-ImageNet','CKA','pretrain_loss','A_score','eff_rank','RankMe','VSA']
sign={'pretrain_loss':-1}
print('=== [A] Coverage confound: each metric on ITS pool vs on the COMMON pool ===')
common=D.dropna(subset=mets+['MLLM_Avg'])
print('common pool n=%d families=%s'%(len(common),sorted(common.family.unique())))
print('%-20s %6s %8s | %6s %8s'%('metric','n_own','rho_own','n_com','rho_com'))
for m in mets:
    s=sign.get(m,1)
    own=D.dropna(subset=[m,'MLLM_Avg'])
    print('%-20s %6d %+8.3f | %6d %+8.3f'%(m,len(own),s*spearmanr(own[m],own.MLLM_Avg).statistic,
          len(common),s*spearmanr(common[m],common.MLLM_Avg).statistic))

print('\n=== [B] which families does each metric even cover? ===')
for m in mets:
    own=D.dropna(subset=[m,'MLLM_Avg'])
    print('%-20s n=%2d  ssl=%d/%d families=%s'%(m,len(own),(own.sup=='ssl').sum(),(D.sup=='ssl').sum(),','.join(sorted(own.family.unique()))))

print('\n=== [C] one-per-family + within-family for each metric (own pool) ===')
rng=np.random.default_rng(0)
print('%-20s %6s %8s %10s %10s'%('metric','n','rho_all','rho_1perfam','rho_within'))
for m in mets:
    s=sign.get(m,1); own=D.dropna(subset=[m,'MLLM_Avg']).copy()
    fams=list(own.family.unique()); rs=[]
    for _ in range(4000):
        idx=[rng.choice(own.index[own.family==f]) for f in fams]
        sub=own.loc[idx]; rs.append(spearmanr(sub[m],sub.MLLM_Avg).statistic)
    own['xc']=own[m]-own.groupby('family')[m].transform('mean')
    own['yc']=own.MLLM_Avg-own.groupby('family').MLLM_Avg.transform('mean')
    big=own[own.family.map(own.family.value_counts())>=3]
    rw=spearmanr(big.xc,big.yc).statistic if len(big)>5 else np.nan
    print('%-20s %6d %+8.3f %+10.3f %+10.3f'%(m,len(own),s*spearmanr(own[m],own.MLLM_Avg).statistic,s*np.mean(rs),s*rw))

print('\n=== [D] LLM dependence of the probing residual ===')
for col in ['qwen3','qwen2_5','smollm2']:
    sub=D.dropna(subset=[col,'probe_epoch1'])
    A=np.polyfit(sub.probe_epoch1,sub[col],1); r=sub[col]-np.polyval(A,sub.probe_epoch1)
    sub=sub.assign(r=r)
    ss=sum(len(g)*(g.r.mean()-r.mean())**2 for _,g in sub.groupby('family'))/((r-r.mean())**2).sum()
    print('\n%s: n=%d  rho=%.3f  slope=%.3f  family-explains-residual=%.3f'%(col,len(sub),spearmanr(sub.probe_epoch1,sub[col]).statistic,A[0],ss))
    fr=sub.groupby('family').r.agg(['mean','count'])
    fr=fr[fr['count']>=2].sort_values('mean')
    print('  '+' | '.join('%s %+.1f(n%d)'%(k,v['mean'],v['count']) for k,v in fr.iterrows()))
