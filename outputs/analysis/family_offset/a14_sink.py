"""How much of the POOLED vector that every cheap benchmark scores is contributed by a
handful of outlier tokens?  If it is most of it, the benchmark is measuring sinks, not the image."""
import sys, yaml, numpy as np, torch
sys.path.insert(0,'/cache/ma-user/VTBenchLab'); sys.path.insert(0,'/cache/ma-user/VTBenchLab/lar')
from pathlib import Path
import model_adapters as MA
from data import image_path
from torch.utils.data import DataLoader, Dataset
from PIL import Image
_o=MA.PatchMeanEncoder._finish
def _f(self,t):
    self.last_tokens=t.detach().float(); return _o(self,t)
MA.PatchMeanEncoder._finish=_f
R=Path('/cache/ma-user/VTBenchLab')
ids=[l.strip() for l in open(R/'lar/features/dinov2_large__coco4618.ids.txt')]
rng=np.random.default_rng(0); sel=[ids[i] for i in rng.permutation(len(ids))[:128]]
class DS(Dataset):
    def __init__(s,i,t): s.i=i; s.t=t
    def __len__(s): return len(s.i)
    def __getitem__(s,k):
        with Image.open(image_path('coco4618',s.i[k])) as im: return s.t(im.convert('RGB'))
specs={r['name']:r for r in yaml.safe_load(open(R/'lar/configs/models_e3.yaml'))['models']}
dev=torch.device('cuda')
for nm in ['pe_lang_g14_448','pe_core_g14_448','pe_lang_l14_448','siglip2_g16_384','dinov2_large','dino_vitb16','mc2_l14_224']:
    b=MA.load_patch_bundle(specs[nm]['loader_name'],dev)
    dl=DataLoader(DS(sel,b.eval_transform),batch_size=8,num_workers=4)
    fr=[];sh=[];topfrac=[]
    for imgs in dl:
        with torch.inference_mode(), b.autocast_context(): b.encoder(imgs.to(dev))
        T=b.encoder.last_tokens.double()           # [B,N,D]
        n=T.norm(dim=-1)                            # [B,N]
        med=n.median(1,keepdim=True).values
        out=(n>5*med)                               # outlier tokens
        fr.append(float(out.float().mean(1).mean()))
        pooled=T.mean(1)
        contrib=(T*out[...,None]).sum(1)/T.shape[1]     # part of the pooled vector from outliers
        sh.append(float((contrib.norm(dim=-1)/(pooled.norm(dim=-1)+1e-9)).mean()))
        k=max(1,T.shape[1]//100)
        idx=n.topk(k,dim=1).indices
        top=torch.gather(T,1,idx[...,None].expand(-1,-1,T.shape[-1])).sum(1)/T.shape[1]
        topfrac.append(float((top.norm(dim=-1)/(pooled.norm(dim=-1)+1e-9)).mean()))
    print('%-20s tokens=%4d | outlier(>5x med) share of tokens=%.4f | ||pooled part from outliers||/||pooled||=%.3f | top-1%% tokens contribute %.3f of pooled norm'%(
        nm,T.shape[1],np.mean(fr),np.mean(sh),np.mean(topfrac))); sys.stdout.flush()
    del b; torch.cuda.empty_cache()
