"""Score SAIL alignment probing against MLLM under this project's three protocols."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
R='/cache/ma-user/VTBenchLab/'
A=pd.read_csv(R+'outputs/analysis/alignment_probing/alignment_scores.csv')
A['align_R1']=0.5*(A['test_image_to_text_R@1']+A['test_text_to_image_R@1'])
A['align_negrank']=-0.5*(A['test_image_to_text_mean_rank']+A['test_text_to_image_mean_rank'])
T=pd.read_csv(R+'lar/configs/e3_targets.csv')
G=pd.read_csv(R+'outputs/analysis/readout_protocol/readout_labels.csv')
M=T.merge(A,on='name',how='inner').merge(G,on='name',how='left').dropna(subset=['MLLM_Avg']).reset_index(drop=True)
print('alignment 跑完 %d 个; 与 MLLM 可比 %d 个, 族 %d 个\n'%(len(A),len(M),M.family.nunique()))
rng=np.random.default_rng(0)
def protocols(col,d,tag):
    d=d.dropna(subset=[col]).reset_index(drop=True).copy(); d['s']=d[col].astype(float)
    fs=list(d.family.unique()); rs=[];rg=[]
    for _ in range(4000):
        g=d.loc[[rng.choice(d.index[d.family==f]) for f in fs]]; rs.append(spearmanr(g.s,g.MLLM_Avg).statistic)
    for _ in range(8000):
        pick=rng.choice(len(fs),min(5,len(fs)),replace=False)
        g=d.loc[[rng.choice(d.index[d.family==fs[j]]) for j in pick]]
        rg.append(g.MLLM_Avg.max()-g.MLLM_Avg.values[g.s.values.argmax()])
    top=d[d.MLLM_Avg>=d.MLLM_Avg.quantile(.75)]
    P=np.polyfit(d.s,d.MLLM_Avg,1); r=d.MLLM_Avg-np.polyval(P,d.s)
    ss=sum(len(g)*(g.mean()-r.mean())**2 for _,g in r.groupby(d.family))/((r-r.mean())**2).sum()
    print('%-38s %4d %+6.3f %+8.3f %8.2f %+8.3f %8.3f'%(tag,len(d),spearmanr(d.s,d.MLLM_Avg).statistic,
        np.mean(rs),np.mean(rg),spearmanr(top.s,top.MLLM_Avg).statistic,ss))
print('%-38s %4s %6s %8s %8s %8s %8s'%('score','n','rho','1perfam','regret','top25','famres'))
protocols('align_R1',M,'SAIL alignment probing (R@1)')
protocols('align_negrank',M,'SAIL alignment probing (−mean rank)')
protocols('ridge_std_obj80',M,'ridge+标准化+多标签 (目前最好)')
protocols('probe_epoch1',M,'ImageNet probing')
protocols('A_score',M,'A score')
print('\n=== alignment 与其它分数的一致性 ===')
for c in ['ridge_std_obj80','probe_epoch1','A_score']:
    s=M.dropna(subset=['align_R1',c])
    print('  align_R1 vs %-18s %+0.3f  (n=%d)'%(c,spearmanr(s.align_R1,s[c]).statistic,len(s)))
print('\n=== alignment 分数最高/最低各 5 ===')
d=M.sort_values('align_R1',ascending=False)
print(d.head(5)[['name','family','MLLM_Avg','align_R1']].round(3).to_string(index=False))
print(d.tail(5)[['name','family','MLLM_Avg','align_R1']].round(3).to_string(index=False))
