"""Final battery: COCO scene read-outs + TextVQA text read-outs. Fixed, pre-specified weights
(equal-weight z-score) so the score is inductive - a single new tokenizer can be scored alone."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
S='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/'
M=pd.read_csv(S+'battery_merged.csv')
O=pd.read_csv(S+'ocr_probes.csv')
M=M.merge(O,on='name',how='left',suffixes=('','_o'))
print('n=%d  有 OCR readout 的: %d'%(len(M),M.mAP_ocr_word.notna().sum()))
rng=np.random.default_rng(0)
def protocols(s,d,tag,B=400):
    d=d.reset_index(drop=True).copy(); d['s']=np.asarray(s,float); d=d.dropna(subset=['s','MLLM_Avg']).reset_index(drop=True)
    fs=list(d.family.unique()); rs=[];rg=[]
    for _ in range(4000):
        idx=[rng.choice(d.index[d.family==f]) for f in fs]; g=d.loc[idx]
        rs.append(spearmanr(g.s,g.MLLM_Avg).statistic)
    for _ in range(8000):
        pick=rng.choice(len(fs),min(5,len(fs)),replace=False)
        idx=[rng.choice(d.index[d.family==fs[j]]) for j in pick]; g=d.loc[idx]
        rg.append(g.MLLM_Avg.max()-g.MLLM_Avg.values[g.s.values.argmax()])
    top=d[d.MLLM_Avg>=d.MLLM_Avg.quantile(.75)]
    # bootstrap CI on full-table rho
    br=[spearmanr(d.s.values[i],d.MLLM_Avg.values[i]).statistic for i in (rng.integers(0,len(d),(B,len(d))))]
    return dict(tag=tag,n=len(d),rho=spearmanr(d.s,d.MLLM_Avg).statistic,rho_lo=np.percentile(br,2.5),rho_hi=np.percentile(br,97.5),
                onefam=np.mean(rs),ofsd=np.std(rs),regret=np.mean(rg),top25=spearmanr(top.s,top.MLLM_Avg).statistic,ntop=len(top))
def show(r): print('%-40s %4d %+6.3f [%+.2f,%+.2f] %+7.3f %8.2f %+8.3f'%(r['tag'],r['n'],r['rho'],r['rho_lo'],r['rho_hi'],r['onefam'],r['regret'],r['top25']))
def z(cols,d=None):
    d=M if d is None else d
    return ((d[cols]-d[cols].mean())/d[cols].std()).mean(1)

OCR=[c for c in M.columns if c.startswith('mAP_ocr')]
print('\n=== 新增 readout 单项 ===')
print('%-40s %4s %6s %14s %7s %8s %8s'%('score','n','rho','95%CI','1perfam','regret','top25'))
for c in OCR+['mAP_oi_class']:
    if c in M: show(protocols(M[c],M,c))
print('\n=== 电池组合（等权 z-score，固定权重，可归纳）===')
CFG={
 'VTB-4  scene only (obj80,small,crowd,w_act)':['mAP_obj80','mAP_small','mAP_crowd','mAP_w_act'],
 'VTB-5  + ocr_word':['mAP_obj80','mAP_small','mAP_crowd','mAP_w_act','mAP_ocr_word'],
 'VTB-6  + ocr_word + ocr_dense':['mAP_obj80','mAP_small','mAP_crowd','mAP_w_act','mAP_ocr_word','mAP_ocr_dense'],
 'VTB-5  + oi_class  (对照:同图非OCR)':['mAP_obj80','mAP_small','mAP_crowd','mAP_w_act','mAP_oi_class'],
 'VTB-6  + probing':['mAP_obj80','mAP_small','mAP_crowd','mAP_w_act','mAP_ocr_word','probe_epoch1'],
 'VTB-7  + ocr_dense + probing':['mAP_obj80','mAP_small','mAP_crowd','mAP_w_act','mAP_ocr_word','mAP_ocr_dense','probe_epoch1'],
}
for nm,c in CFG.items():
    if not all(x in M for x in c): continue
    sub=M.dropna(subset=c); show(protocols(z(c,sub),sub,nm))
print('\n=== 对照 ===')
for c,sg,nm in [('probe_epoch1',1,'ImageNet probing (baseline)'),('A_score',1,'A score (Law of Vision Rep.)'),
                ('mean_domain_rank',-1,'gated loss proxy')]:
    if c in M:
        sub=M.dropna(subset=[c]).reset_index(drop=True); show(protocols(sg*sub[c].values,sub,nm))
best=['mAP_obj80','mAP_small','mAP_crowd','mAP_w_act','mAP_ocr_word','mAP_ocr_dense','probe_epoch1']
if all(c in M for c in best):
    sub=M.dropna(subset=best).reset_index(drop=True); s=z(best,sub)
    A=np.polyfit(s,sub.MLLM_Avg,1); r=sub.MLLM_Avg-np.polyval(A,s)
    sub=sub.assign(r=r)
    ss=sum(len(g)*(g.r.mean()-r.mean())**2 for _,g in sub.groupby('family'))/((r-r.mean())**2).sum()
    print('\n最终电池的残差被家族解释: %.3f   (probing 0.806 / loss proxy 0.674)'%ss)
    print(sub.groupby('family').r.agg(['mean','count']).sort_values('mean').round(2).to_string())
M.to_csv(S+'final_merged.csv',index=False)
