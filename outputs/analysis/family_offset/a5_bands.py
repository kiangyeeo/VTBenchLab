import pandas as pd, numpy as np
from scipy.stats import spearmanr
D=pd.read_csv('/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/master.csv')
D=D.dropna(subset=['MLLM_Avg','probe_epoch1'])
print('=== [E] MLLM spread inside narrow ImageNet-probing bands ===')
for w in [1.0,2.0,3.0]:
    best=None
    for c in np.arange(D.probe_epoch1.min(),D.probe_epoch1.max(),0.25):
        s=D[(D.probe_epoch1>=c)&(D.probe_epoch1<c+w)]
        if len(s)>=4:
            sp=s.MLLM_Avg.max()-s.MLLM_Avg.min()
            if best is None or sp>best[0]: best=(sp,c,s)
    sp,c,s=best
    print('\nwidth=%.1f  band probe∈[%.1f,%.1f]  n=%d  MLLM spread=%.1f pts (target total range=%.1f)'%(w,c,c+w,len(s),sp,D.MLLM_Avg.max()-D.MLLM_Avg.min()))
    print(s[['name','family','probe_epoch1','MLLM_Avg']].sort_values('MLLM_Avg').to_string(index=False))

print('\n=== [F] controlled pairs: matched arch/resolution, different objective ===')
pairs=[('pe_core_g14_448','pe_lang_g14_448','same PE g14@448, CLIP-contrastive vs language-aligned'),
       ('dinov2_large','mc2_l14_224','probe-matched, SSL vs lang'),
       ('raev2','mc2_l14_224','probe-matched, SSL vs lang'),
       ('dinov3','dinov2_large','same lab, v3 vs v2'),
       ('dino_vitb8','dino_vitb16','same DINO ViT-B, patch 8 vs 16 (784 vs 196 tokens)'),
       ('dino_vits8','dino_vits16','same DINO ViT-S, patch 8 vs 16'),
       ('webssl_mae300m_full2b_224','dino_vits16','MAE vs DINO, both pure SSL'),
       ('siglip2_b16_224','siglip2_b16_512','same siglip2 B/16, res 224 vs 512')]
idx=D.set_index('name')
for a,b,why in pairs:
    if a in idx.index and b in idx.index:
        ra,rb=idx.loc[a],idx.loc[b]
        print('%-28s vs %-28s  Δprobe=%+6.2f  ΔMLLM=%+6.2f   %s'%(a,b,rb.probe_epoch1-ra.probe_epoch1,rb.MLLM_Avg-ra.MLLM_Avg,why))

print('\n=== [G] how many pairs does probing rank BACKWARDS, by pair type ===')
n=len(D); rows=D.reset_index(drop=True)
same=0;samew=0;diff=0;diffw=0
for i in range(n):
    for j in range(i+1,n):
        a,b=rows.iloc[i],rows.iloc[j]
        if abs(a.MLLM_Avg-b.MLLM_Avg)<1.0: continue
        wrong=(a.probe_epoch1-b.probe_epoch1)*(a.MLLM_Avg-b.MLLM_Avg)<0
        if a.family==b.family: same+=1; samew+=wrong
        else: diff+=1; diffw+=wrong
print('within-family pairs : %d, discordant %d (%.1f%%)'%(same,samew,100*samew/same))
print('cross-family pairs  : %d, discordant %d (%.1f%%)'%(diff,diffw,100*diffw/diff))
