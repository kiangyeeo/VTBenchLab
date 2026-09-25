import pandas as pd, numpy as np
from scipy.stats import spearmanr, pearsonr
S='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/'
D=pd.read_csv(S+'merged.csv')
print('=== [H] intercept-only vs slope-varying family model (probe -> MLLM) ===')
big=D[D.family.map(D.family.value_counts())>=4].copy()
import itertools
from sklearn.linear_model import LinearRegression
X0=big[['probe_epoch1']].values; y=big.MLLM_Avg.values
Fd=pd.get_dummies(big.family,drop_first=True).astype(float).values
def r2(Xm):
    m=LinearRegression().fit(Xm,y); return 1-((y-m.predict(Xm))**2).sum()/((y-y.mean())**2).sum(), Xm.shape[1]
for nm,Xm in [('probe only',X0),('family only',Fd),('probe+family (shared slope)',np.hstack([X0,Fd])),
              ('probe*family (own slope)',np.hstack([X0,Fd,Fd*X0]))]:
    r,k=r2(Xm); print('  %-30s R2=%.3f  (k=%d params)'%(nm,r,k))
# per-family slopes
print('\n  per-family slope of MLLM vs probe:')
for f,g in big.groupby('family'):
    if len(g)>=4:
        s=np.polyfit(g.probe_epoch1,g.MLLM_Avg,1)
        print('    %-12s n=%2d slope=%+.3f  intercept-at-probe80=%.1f'%(f,len(g),s[0],np.polyval(s,80)))

print('\n=== [I] is the family offset LLM-invariant? (models scored by >=2 LLMs) ===')
sub=D.dropna(subset=['qwen2_5','smollm2','probe_epoch1'])
print('n(qwen2.5 & smollm2) =',len(sub))
for c in ['qwen2_5','smollm2']:
    z=(sub[c]-sub[c].mean())/sub[c].std(); sub[c+'_z']=z
A=np.polyfit(sub.probe_epoch1,sub.qwen2_5_z,1); sub['r_q25']=sub.qwen2_5_z-np.polyval(A,sub.probe_epoch1)
A=np.polyfit(sub.probe_epoch1,sub.smollm2_z,1); sub['r_sml']=sub.smollm2_z-np.polyval(A,sub.probe_epoch1)
print('per-model residual correlation across the two LLMs: pearson=%.3f spearman=%.3f'%(
    pearsonr(sub.r_q25,sub.r_sml).statistic,spearmanr(sub.r_q25,sub.r_sml).statistic))
fr=sub.groupby('family')[['r_q25','r_sml']].mean().join(sub.groupby('family').size().rename('n'))
print(fr[fr.n>=2].round(2).sort_values('r_q25').to_string())
sub2=D.dropna(subset=['qwen2_5','qwen3','probe_epoch1']).copy()
for c in ['qwen2_5','qwen3']: sub2[c+'_z']=(sub2[c]-sub2[c].mean())/sub2[c].std()
A=np.polyfit(sub2.probe_epoch1,sub2.qwen2_5_z,1); a=sub2.qwen2_5_z-np.polyval(A,sub2.probe_epoch1)
A=np.polyfit(sub2.probe_epoch1,sub2.qwen3_z,1); b=sub2.qwen3_z-np.polyval(A,sub2.probe_epoch1)
print('\nqwen2.5 vs qwen3 residual agreement (n=%d): pearson=%.3f spearman=%.3f'%(len(sub2),pearsonr(a,b).statistic,spearmanr(a,b).statistic))

print('\n=== [J] what A_score is really doing ===')
d=D.dropna(subset=['A_score'])
print('A_score vs MLLM rho=%.3f | vs probe rho=%.3f | vs probing-residual rho=%.3f'%(
    spearmanr(d.A_score,d.MLLM_Avg).statistic,spearmanr(d.A_score,d.probe_epoch1).statistic,spearmanr(d.A_score,d.res).statistic))
print('A_score vs family-mean-offset: ', spearmanr(d.groupby('family').A_score.mean(),d.groupby('family').res.mean()).statistic.round(3))
print('families A_score covers   :',sorted(d.family.unique()))
print('families A_score MISSES   :',sorted(set(D.family)-set(d.family)))
print('\nA_score within-family spearman:')
for f,g in d.groupby('family'):
    if len(g)>=4: print('   %-12s n=%2d rho=%+.3f'%(f,len(g),spearmanr(g.A_score,g.MLLM_Avg).statistic))
