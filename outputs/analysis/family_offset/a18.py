import pandas as pd, numpy as np
from scipy.stats import spearmanr, pearsonr
S='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/'
E=pd.read_csv('/cache/ma-user/VTBenchLab/gradient_compatibility/artifacts/full_sweep_v1/summary/evaluation.csv')
E['name']=E.registry_name
D=pd.read_csv(S+'merged.csv'); C=pd.read_csv(S+'coco_probes.csv')
M=E.merge(D[['name','probe_epoch1','res']],on='name').merge(C,on='name')
M['family']=M['family_x'] if 'family_x' in M else M['family']
M['cap']=-M.caption_loss
A=np.polyfit(M.cap,M.qwen2_5,1); M['res_cap']=M.qwen2_5-np.polyval(A,M.cap)
print('caption_loss 与 probing 的相关: spearman=%.3f'%spearmanr(M.cap,M.probe_epoch1).statistic)
print('两者残差的相关（失效是否同向）: pearson=%.3f spearman=%.3f'%(pearsonr(M.res_cap,M.res).statistic,spearmanr(M.res_cap,M.res).statistic))
ssb=sum(len(g)*(g.res_cap.mean()-M.res_cap.mean())**2 for _,g in M.groupby('family'))/((M.res_cap-M.res_cap.mean())**2).sum()
print('caption 代理的残差被家族解释: %.3f  (probing 是 0.806)'%ssb)
print('\ncaption 代理的家族偏移:')
print(M.groupby('family').res_cap.agg(['mean','count']).sort_values('mean').round(2).to_string())
