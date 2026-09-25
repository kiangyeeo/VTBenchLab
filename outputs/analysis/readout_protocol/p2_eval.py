import numpy as np, pandas as pd
from scipy.stats import spearmanr
S='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/da5d0c83-f13a-456d-b6e7-8b458f5e0d91/scratchpad/'
R='/cache/ma-user/VTBenchLab/'
X=pd.read_csv(R+'outputs/analysis/readout_protocol/readout_2x2.csv')
B=pd.read_csv(S+'coco_vs_in.csv')[['name','family','MLLM_Avg','probe_epoch1','coco_mAP']]
M=B.merge(X,on='name',how='inner').dropna(subset=['MLLM_Avg']).reset_index(drop=True)
print('2x2 跑完: %d 个 encoder; 与 MLLM + 你的 SGD 版可比的: %d  families=%d'%(len(X),len(M),M.family.nunique()))
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
    A=np.polyfit(d.s,d.MLLM_Avg,1); r=d.MLLM_Avg-np.polyval(A,d.s)
    ss=sum(len(g)*(g.mean()-r.mean())**2 for _,g in r.groupby(d.family))/((r-r.mean())**2).sum()
    print('%-44s %4d %+6.3f %+8.3f %8.2f %+8.3f %8.3f'%(tag,len(d),spearmanr(d.s,d.MLLM_Avg).statistic,
        np.mean(rs),np.mean(rg),spearmanr(top.s,top.MLLM_Avg).statistic,ss))
print('\n=== 全部在同一批 %d 个模型上 ==='%len(M))
print('%-44s %4s %6s %8s %8s %8s %8s'%('score','n','rho','1perfam','regret','top25','famres'))
for c,t in [('ridge_std','A ridge + 逐维标准化      (=a7 mAP_obj80)'),
            ('ridge_raw','B ridge + 只中心化'),
            ('sgd_std','C SGD+13LR + 逐维标准化'),
            ('sgd_raw','D SGD+13LR + 只中心化   (=协议的 feature_norm=False)'),
            ('coco_mAP','E 你的: SGD+13LR + 活backbone+增强 + 82k'),
            ('probe_epoch1','F ImageNet probing 原版')]:
    if c in M: protocols(c,M,t)
print('\n=== 分数之间的一致性 (Spearman) ===')
for a,b in [('ridge_std','ridge_raw'),('ridge_std','sgd_std'),('sgd_std','sgd_raw'),('sgd_raw','coco_mAP'),('ridge_std','coco_mAP')]:
    print('  %-24s vs %-12s %+0.3f'%(a,b,spearmanr(M[a],M[b]).statistic))
