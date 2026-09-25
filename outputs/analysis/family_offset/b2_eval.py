"""Honest evaluation of the readout battery. Every reported number for a combined score comes
from leave-one-FAMILY-out predictions, so no model ever sees its own family."""
import pandas as pd, numpy as np, itertools, sys
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
S='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/'
B=pd.read_csv(S+'battery.csv'); D=pd.read_csv(S+'merged.csv')
E=pd.read_csv('/cache/ma-user/VTBenchLab/gradient_compatibility/artifacts/full_sweep_v1/summary/evaluation.csv')
E['name']=E.registry_name
M=D.merge(B,on='name').merge(E[['name','mean_domain_rank','caption_loss']],on='name',how='left')
M=M.dropna(subset=['MLLM_Avg','probe_epoch1']).reset_index(drop=True)
READ=[c for c in B.columns if c.startswith('mAP_')]
print('n=%d families=%d  readouts=%s'%(len(M),M.family.nunique(),READ))
rng=np.random.default_rng(0)

def protocols(score,df,sign=1,tag=''):
    d=df.dropna(subset=['MLLM_Avg']).copy(); d['s']=sign*np.asarray(score,float)
    d=d.dropna(subset=['s'])
    rho=spearmanr(d.s,d.MLLM_Avg).statistic
    fs=list(d.family.unique()); rs=[];rg=[]
    for _ in range(4000):
        idx=[rng.choice(d.index[d.family==f]) for f in fs]; g=d.loc[idx]
        rs.append(spearmanr(g.s,g.MLLM_Avg).statistic)
    for _ in range(8000):
        pick=rng.choice(len(fs),min(5,len(fs)),replace=False)
        idx=[rng.choice(d.index[d.family==fs[j]]) for j in pick]; g=d.loc[idx]
        rg.append(g.MLLM_Avg.max()-g.MLLM_Avg.values[g.s.values.argmax()])
    top=d[d.MLLM_Avg>=d.MLLM_Avg.quantile(.75)]
    return dict(tag=tag,n=len(d),rho=rho,onefam=np.mean(rs),regret=np.mean(rg),
                top25=spearmanr(top.s,top.MLLM_Avg).statistic)

def lofo_pred(df,feats):
    d=df.dropna(subset=feats+['MLLM_Avg']).copy()
    out=pd.Series(np.nan,index=d.index)
    for f in d.family.unique():
        tr=d[d.family!=f]; te=d[d.family==f]
        if len(tr)<10: continue
        mu,sd=tr[feats].mean(),tr[feats].std()+1e-9
        m=RidgeCV(alphas=np.logspace(-2,4,25)).fit((tr[feats]-mu)/sd,tr.MLLM_Avg)
        out.loc[te.index]=m.predict((te[feats]-mu)/sd)
    return d,out

print('\n=== 单项 readout ===')
print('%-14s %6s %7s %8s %8s %8s'%('score','n','rho','1perfam','regret','top25'))
res=[]
for c in READ+['probe_epoch1','A_score','mean_domain_rank','caption_loss']:
    if c not in M: continue
    sg=-1 if c in ('mean_domain_rank','caption_loss') else 1
    r=protocols(M[c],M,sg,c); res.append(r)
    print('%-14s %6d %+7.3f %+8.3f %8.2f %+8.3f'%(c,r['n'],r['rho'],r['onefam'],r['regret'],r['top25']))

print('\n=== 组合分（leave-one-FAMILY-out 预测，权重从不接触本家族） ===')
COMBOS={
 'battery-all':READ,
 'battery+probe':READ+['probe_epoch1'],
 'core4':['mAP_obj80','mAP_small','mAP_w_obj','mAP_w_rel'],
 'core4+probe':['mAP_obj80','mAP_small','mAP_w_obj','mAP_w_rel','probe_epoch1'],
 'core6':['mAP_obj80','mAP_small','mAP_count2','mAP_w_obj','mAP_w_rel','mAP_w_color'],
 'core6+probe':['mAP_obj80','mAP_small','mAP_count2','mAP_w_obj','mAP_w_rel','mAP_w_color','probe_epoch1'],
 'probe-only':['probe_epoch1'],
}
print('%-16s %6s %7s %8s %8s %8s'%('score','n','rho','1perfam','regret','top25'))
best=None
for nm,f in COMBOS.items():
    d,p=lofo_pred(M,f); ok=p.notna()
    r=protocols(p[ok],d[ok],1,nm)
    print('%-16s %6d %+7.3f %+8.3f %8.2f %+8.3f'%(nm,r['n'],r['rho'],r['onefam'],r['regret'],r['top25']))
    if best is None or r['onefam']>best[1]['onefam']: best=(nm,r,f)

print('\n=== 无需拟合的固定分（z-score 等权平均，可归纳，n=1 也能算） ===')
print('%-34s %6s %7s %8s %8s %8s'%('score','n','rho','1perfam','regret','top25'))
for nm,f in [('obj80+small','mAP_obj80 mAP_small'),
             ('obj80+small+w_obj','mAP_obj80 mAP_small mAP_w_obj'),
             ('obj80+small+w_obj+w_rel','mAP_obj80 mAP_small mAP_w_obj mAP_w_rel'),
             ('obj80+small+w_obj+w_rel+count2','mAP_obj80 mAP_small mAP_w_obj mAP_w_rel mAP_count2'),
             ('+probe','mAP_obj80 mAP_small mAP_w_obj mAP_w_rel probe_epoch1')]:
    cols=f.split(); Zs=((M[cols]-M[cols].mean())/M[cols].std()).mean(1)
    r=protocols(Zs,M,1,nm)
    print('%-34s %6d %+7.3f %+8.3f %8.2f %+8.3f'%(nm,r['n'],r['rho'],r['onefam'],r['regret'],r['top25']))

print('\n=== 最佳组合 (%s) 的家族残差 ==='%best[0])
d,p=lofo_pred(M,best[2]); ok=p.notna(); d=d[ok].copy(); d['pred']=p[ok]
d['r']=d.MLLM_Avg-d.pred
ss=sum(len(g)*(g.r.mean()-d.r.mean())**2 for _,g in d.groupby('family'))/((d.r-d.r.mean())**2).sum()
print('残差被家族解释: %.3f   (probing 是 0.806, caption loss 代理是 0.674)'%ss)
print(d.groupby('family').r.agg(['mean','count']).sort_values('mean').round(2).to_string())
M.to_csv(S+'battery_merged.csv',index=False)
