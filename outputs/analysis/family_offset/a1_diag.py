import pandas as pd, numpy as np
from scipy.stats import spearmanr, pearsonr
pd.set_option('display.width',200)

R='/cache/ma-user/VTBenchLab/'
t=pd.read_csv(R+'lar/configs/e3_targets.csv')
lm=pd.read_csv(R+'lar/results/lar_metrics_v2.csv')
lm=lm[(lm.text_domain=='caption')&(lm.image_set=='coco4618')][['name','d','n_tokens','eff_rank','RankMe','VSA','m50','m90','LAR_64']]
df=t.merge(lm,on='name',how='left')

# supervision taxonomy
sup={'siglip2':'lang','mc1':'lang','mc2':'lang','clip':'lang','toklip':'lang','unitok':'lang','vilau':'lang',
     'pe_core':'lang','pe_lang':'lang','eupe':'lang',
     'dino':'ssl','dinov2':'ssl','dinov3':'ssl','ijepa':'ssl','webssl_mae':'ssl','pixio':'ssl','raev2':'ssl','uniar':'other'}
df['sup']=df.family.map(sup)
# webssl_dino rows are in family 'dino'
df.loc[df.name.str.startswith('webssl_dino'),'sup']='ssl'

D=df.dropna(subset=['MLLM_Avg','probe_epoch1']).copy()
print('n with target+probe:',len(D))
print('\n=== [1] Overall vs per-family Spearman (probe -> MLLM_Avg) ===')
print('overall rho=%.3f (n=%d)'%(spearmanr(D.probe_epoch1,D.MLLM_Avg).statistic,len(D)))
for s in ['lang','ssl']:
    sub=D[D.sup==s]
    print(' %-5s rho=%.3f n=%d'%(s,spearmanr(sub.probe_epoch1,sub.MLLM_Avg).statistic,len(sub)))
print('\nper-family:')
for f,g in D.groupby('family'):
    if len(g)>=3:
        print('  %-12s n=%2d rho=%+.3f  probe[%.1f,%.1f] mllm[%.1f,%.1f]'%(f,len(g),spearmanr(g.probe_epoch1,g.MLLM_Avg).statistic,g.probe_epoch1.min(),g.probe_epoch1.max(),g.MLLM_Avg.min(),g.MLLM_Avg.max()))

print('\n=== [2] one-per-family bootstrap ===')
rng=np.random.default_rng(0)
fams=list(D.family.unique())
rs=[]
for _ in range(20000):
    idx=[rng.choice(D.index[D.family==f]) for f in fams]
    s=D.loc[idx]
    rs.append(spearmanr(s.probe_epoch1,s.MLLM_Avg).statistic)
print('one-per-family rho = %.3f +- %.3f (nfam=%d)'%(np.mean(rs),np.std(rs),len(fams)))

print('\n=== [3] residual structure: MLLM ~ a*probe+b, residual by family ===')
x=D.probe_epoch1.values; y=D.MLLM_Avg.values
A=np.polyfit(x,y,1); pred=np.polyval(A,x); res=y-pred
D['res']=res
print('global fit: MLLM = %.3f*probe + %.2f ; R2=%.3f'%(A[0],A[1],pearsonr(x,y).statistic**2))
fam_res=D.groupby('family').res.agg(['mean','std','count']).sort_values('mean')
print(fam_res.round(2))
# variance decomposition of residual
grand=res.mean()
ss_tot=((res-grand)**2).sum()
ss_between=sum(len(g)*(g.res.mean()-grand)**2 for _,g in D.groupby('family'))
print('\nresidual variance explained by FAMILY identity: %.3f'%(ss_between/ss_tot))
# same for raw target
ss_tot_y=((y-y.mean())**2).sum()
ss_b_y=sum(len(g)*(g.MLLM_Avg.mean()-y.mean())**2 for _,g in D.groupby('family'))
print('target variance explained by FAMILY identity  : %.3f'%(ss_b_y/ss_tot_y))

print('\n=== [4] within-family centered correlation (remove family mean from both) ===')
Dw=D.copy()
Dw['p_c']=Dw.probe_epoch1-Dw.groupby('family').probe_epoch1.transform('mean')
Dw['y_c']=Dw.MLLM_Avg-Dw.groupby('family').MLLM_Avg.transform('mean')
big=Dw[Dw.family.map(Dw.family.value_counts())>=3]
print('within-family pooled (n=%d): pearson=%.3f spearman=%.3f'%(len(big),pearsonr(big.p_c,big.y_c).statistic,spearmanr(big.p_c,big.y_c).statistic))
# between-family: family means
fm=D.groupby('family').agg(p=('probe_epoch1','mean'),y=('MLLM_Avg','mean'),n=('name','count'))
fm=fm[fm.n>=2]
print('between-family (family means, n=%d): pearson=%.3f spearman=%.3f'%(len(fm),pearsonr(fm.p,fm.y).statistic,spearmanr(fm.p,fm.y).statistic))
print(fm.sort_values('y').round(2))
D.to_csv('/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/master.csv',index=False)
