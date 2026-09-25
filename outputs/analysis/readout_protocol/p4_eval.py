"""Score the label x readout x standardization cube against MLLM under the three protocols."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

R='/cache/ma-user/VTBenchLab/'
X=pd.read_csv(R+'outputs/analysis/readout_protocol/readout_labels.csv')
T=pd.read_csv(R+'lar/configs/e3_targets.csv')[['name','family','MLLM_Avg','probe_epoch1']]
M=X.merge(T,on='name',how='inner').dropna(subset=['MLLM_Avg']).reset_index(drop=True)
print('cube 跑完 %d 个 encoder; 与 MLLM 可比 %d 个, 族 %d 个\n'%(len(X),len(M),M.family.nunique()))

rng=np.random.default_rng(0)
def protocols(col,d):
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
    return dict(rho=spearmanr(d.s,d.MLLM_Avg).statistic,onefam=np.mean(rs),
                regret=np.mean(rg),top25=spearmanr(top.s,top.MLLM_Avg).statistic,famres=ss)

READOUTS=[('ridge_std','ridge + 标准化'),('ridge_raw','ridge + 只中心化'),
          ('sgd_std','SGD + 标准化'),('sgd_raw','SGD + 只中心化')]
res={}
print('%-22s %-10s %6s %8s %8s %8s %8s'%('readout','label','rho','1perfam','regret','top25','famres'))
for key,label in READOUTS:
    for lab in ('obj80','dom'):
        r=protocols(f'{key}_{lab}',M); res[(key,lab)]=r
        print('%-22s %-10s %+6.3f %+8.3f %8.2f %+8.3f %8.3f'%(
            label,'多标签' if lab=='obj80' else '单标签',r['rho'],r['onefam'],r['regret'],r['top25'],r['famres']))
r=protocols('probe_epoch1',M)
print('%-22s %-10s %+6.3f %+8.3f %8.2f %+8.3f %8.3f'%('ImageNet probing 原版','单标签',r['rho'],r['onefam'],r['regret'],r['top25'],r['famres']))

print('\n=== 标签轴的净效应 (多标签 − 单标签，读出固定) ===')
print('%-22s %8s %8s %8s'%('readout','Δ1perfam','Δregret','Δtop25'))
for key,label in READOUTS:
    a,b=res[(key,'obj80')],res[(key,'dom')]
    print('%-22s %+8.3f %+8.2f %+8.3f'%(label,a['onefam']-b['onefam'],a['regret']-b['regret'],a['top25']-b['top25']))
print('\n=== 读出轴的净效应 (ridge+标准化 − SGD+只中心化，标签固定) ===')
for lab in ('obj80','dom'):
    a,b=res[('ridge_std',lab)],res[('sgd_raw',lab)]
    print('%-22s %+8.3f %+8.2f %+8.3f'%('多标签' if lab=='obj80' else '单标签',
          a['onefam']-b['onefam'],a['regret']-b['regret'],a['top25']-b['top25']))
