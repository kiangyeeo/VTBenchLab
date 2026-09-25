import pandas as pd, numpy as np, csv
from scipy.stats import spearmanr
R='/cache/ma-user/VTBenchLab/'
S='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/'
rows=list(csv.reader(open(R+'result/VisualTokenizer表现 - MLLM详细结果 (1).csv',encoding='utf-8-sig')))
tasks=rows[1][2:13]; data=[]
for r in rows[2:]:
    if not r or not r[0]: continue
    d={'name':r[0]}
    for i,t in enumerate(tasks): d['q3|'+t]=float(r[2+i]); d['q25|'+t]=float(r[14+i])
    data.append(d)
T=pd.DataFrame(data)
M=T.merge(pd.read_csv(S+'merged.csv')[['name','family','probe_epoch1','MLLM_Avg','res']],on='name')
C=pd.read_csv(S+'coco_probes.csv'); M=M.merge(C,on='name')
print('n=',len(M))
# z-score each task within each LLM, then average the two LLMs -> a stable 11-dim profile
Z=pd.DataFrame({t: ((M['q3|'+t]-M['q3|'+t].mean())/M['q3|'+t].std() + (M['q25|'+t]-M['q25|'+t].mean())/M['q25|'+t].std())/2 for t in tasks})
U,s,Vt=np.linalg.svd(Z.values-Z.values.mean(0),full_matrices=False)
print('\n=== [P] 11 个任务的因子结构（跨 tokenizer） ===')
print('explained variance ratio:', np.round(s**2/(s**2).sum(),3)[:4])
for k in range(3):
    load=pd.Series(Vt[k],index=tasks).sort_values()
    print('\nPC%d (%.1f%% var)  loadings:'%(k+1,100*s[k]**2/(s**2).sum()))
    print('   '+' '.join('%s=%+.2f'%(t,v) for t,v in load.items()))
F=U[:,:3]*s[:3]
print('\n=== [Q] 每个 cheap 指标分别预测 PC1 / PC2 / PC3 ===')
print('%-16s %8s %8s %8s'%('metric','PC1','PC2','PC3'))
for c in ['probe_epoch1','mAP_obj80','mAP_small','mAP_word_head','mAP_dom']:
    print('%-16s %+8.3f %+8.3f %+8.3f'%(c,*[spearmanr(M[c],F[:,k]).statistic for k in range(3)]))
print('\n=== [R] PC2/PC3 在家族之间的分布（residual 排序） ===')
M2=M.assign(PC1=F[:,0],PC2=F[:,1],PC3=F[:,2])
print(M2.groupby('family').agg(n=('name','count'),res=('res','mean'),PC1=('PC1','mean'),PC2=('PC2','mean'),PC3=('PC3','mean')).sort_values('res').round(2).to_string())
