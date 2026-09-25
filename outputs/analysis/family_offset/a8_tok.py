"""Token-level geometry pilot: every cheap baseline in this project scores a POOLED image vector,
but the MLLM reads a token SEQUENCE. Measure what pooling destroys."""
import sys, json, yaml, numpy as np, torch
sys.path.insert(0,'/cache/ma-user/VTBenchLab')
sys.path.insert(0,'/cache/ma-user/VTBenchLab/lar')
from pathlib import Path
import model_adapters as MA
from data import image_path
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# make PatchMeanEncoder stash the full token tensor
_orig=MA.PatchMeanEncoder._finish
def _finish(self,tokens):
    self.last_tokens=tokens.detach().float()
    return _orig(self,tokens)
MA.PatchMeanEncoder._finish=_finish

R=Path('/cache/ma-user/VTBenchLab')
OUT='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/vtb/'
ids=[l.strip() for l in open(R/'lar/features/dinov2_large__coco4618.ids.txt')]
rng=np.random.default_rng(0); sel=[ids[i] for i in rng.permutation(len(ids))[:512]]

class DS(Dataset):
    def __init__(s,idl,tf): s.i=idl; s.t=tf
    def __len__(s): return len(s.i)
    def __getitem__(s,k):
        with Image.open(image_path('coco4618',s.i[k])) as im: return s.t(im.convert('RGB'))

specs={r['name']:r for r in yaml.safe_load(open(R/'lar/configs/models_e3.yaml'))['models']}
PICK=['dino_vitb16','dino_vitb8','dino_vits16','dinov2_base','dinov2_large','dinov3',
      'webssl_mae300m_full2b_224','webssl_mae3b_full2b_224','I-JEPA','raev2',
      'pixio_vitb16','pixio_vitl16','eupe_vit_b','pe_core_b16_224','pe_core_g14_448',
      'pe_lang_g14_448','pe_lang_l14_448','siglip2_b16_224','siglip2_l16_384','siglip2_g16_384',
      'mc2_l14_224','mc1_l14_224_2.5b','clip_openai__l14','toklip_l_384','unitok_attn','vilau_256']
PICK=[p for p in PICK if p in specs]
print('encoders:',len(PICK)); sys.stdout.flush()
dev=torch.device('cuda')
rows=[]
for nm in PICK:
    sp=specs[nm]
    try:
        b=MA.load_patch_bundle(sp['loader_name'],dev)
    except Exception as e:
        print('LOAD FAIL',nm,repr(e)[:120]); sys.stdout.flush(); continue
    dl=DataLoader(DS(sel,b.eval_transform),batch_size=min(16,int(sp.get('batch_size',16))),num_workers=6)
    acc={'wsum':0.,'wn':0,'gs':[],'er':[],'nr':[],'cos':[]}
    means=[]
    try:
        for imgs in dl:
            imgs=imgs.to(dev,non_blocking=True)
            with torch.inference_mode(), b.autocast_context():
                b.encoder(imgs)
            T=b.encoder.last_tokens                       # [B,T,D]
            if T is None or T.ndim!=3: raise RuntimeError('no tokens')
            m=T.mean(1)                                   # [B,D] image mean == pooled feature
            C=T-m[:,None,:]                               # within-image deviation
            acc['wsum']+=float((C**2).sum()); acc['wn']+=C.shape[0]*C.shape[1]
            means.append(m.cpu())
            # per-image token effective rank (on centred tokens)
            for i in range(T.shape[0]):
                s=torch.linalg.svdvals(C[i].double())
                p=(s**2); p=p/p.sum()
                acc['er'].append(float(torch.exp(-(p*torch.log(p+1e-30)).sum())))
            n=T.norm(dim=-1)                              # [B,T]
            acc['nr'].append(float((n.max(1).values/ (n.median(1).values+1e-9)).mean()))
            Tn=torch.nn.functional.normalize(T,dim=-1)
            acc['cos'].append(float(torch.einsum('btd,bsd->bts',Tn,Tn).mean()))
    except Exception as e:
        print('RUN FAIL',nm,repr(e)[:160]); sys.stdout.flush(); del b; torch.cuda.empty_cache(); continue
    M=torch.cat(means).double()
    between=float(((M-M.mean(0))**2).sum())/M.shape[0]
    within=acc['wsum']/acc['wn']*1.0
    r=dict(name=nm,n_tokens=int(T.shape[1]),d=int(T.shape[2]),
           spatial_share=within/(within+between),
           tok_eff_rank=float(np.mean(acc['er'])),
           tok_eff_rank_frac=float(np.mean(acc['er']))/T.shape[1],
           norm_max_over_med=float(np.mean(acc['nr'])),
           tok_mean_cos=float(np.mean(acc['cos'])))
    rows.append(r); print({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()}); sys.stdout.flush()
    import pandas as pd; pd.DataFrame(rows).to_csv(OUT+'tok_stats.csv',index=False)
    del b; torch.cuda.empty_cache()
print('done')
