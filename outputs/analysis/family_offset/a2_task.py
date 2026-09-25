import pandas as pd, numpy as np, csv
from scipy.stats import spearmanr
R='/cache/ma-user/VTBenchLab/'
rows=list(csv.reader(open(R+'result/VisualTokenizer表现 - MLLM详细结果 (1).csv',encoding='utf-8-sig')))
tasks=rows[1][2:14]  # 11 tasks + Avg
tasks=rows[1][2:13]
data=[]
for r in rows[2:]:
    if not r or not r[0]: continue
    d={'name':r[0]}
    for i,t in enumerate(tasks): d['q3|'+t]=float(r[2+i])
    d['q3|Avg']=float(r[13])
    for i,t in enumerate(tasks): d['q25|'+t]=float(r[14+i])
    d['q25|Avg']=float(r[25])
    data.append(d)
raw=pd.DataFrame(data)
mst=pd.read_csv('/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/master.csv')
m=raw.merge(mst[['name','family','sup','probe_epoch1','res','MLLM_Avg']],on='name',how='inner')
print('merged n=',len(m),' tasks:',tasks)
for pre,llm in [('q3','qwen3 1.7B'),('q25','qwen2.5 1.5B')]:
    print('\n===== %s ====='%llm)
    print('%-10s %6s %6s %6s | %7s %7s %7s'%('task','min','max','sd','rho_avg','rho_prb','rho_res'))
    avg=m[pre+'|Avg']
    for t in tasks:
        v=m[pre+'|'+t]
        print('%-10s %6.1f %6.1f %6.2f | %+7.3f %+7.3f %+7.3f'%(t,v.min(),v.max(),v.std(),
            spearmanr(v,avg).statistic, spearmanr(v,m.probe_epoch1).statistic, spearmanr(v,m.res).statistic))
    V=m[[pre+'|'+t for t in tasks]]
    sh=(V.var()/V.var().sum()).sort_values(ascending=False)
    print('var share of Avg: '+', '.join('%s=%.3f'%(k.split('|')[1],v) for k,v in sh.items()))
