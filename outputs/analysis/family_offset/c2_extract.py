"""Standalone patch-mean feature extraction on the TextVQA image set.
Mirrors lar/extract_visual.py but reads a flat image directory, so no repo file is touched."""
import sys, os, yaml, numpy as np, torch
sys.path.insert(0,'/cache/ma-user/VTBenchLab'); sys.path.insert(0,'/cache/ma-user/VTBenchLab/lar')
import model_adapters as MA
from torch.utils.data import DataLoader, Dataset
from PIL import Image
T='/cache/ma-user/tmp/claude-1000/-cache-ma-user-VTBenchLab/19979757-e891-467a-89b9-7b5340acb66c/scratchpad/textvqa/'
OUT=T+'feat/'; os.makedirs(OUT,exist_ok=True)
ids=[l.strip() for l in open(T+'ids.txt') if l.strip()]
print('images',len(ids)); sys.stdout.flush()
class DS(Dataset):
    def __init__(s,tf): s.tf=tf
    def __len__(s): return len(ids)
    def __getitem__(s,k):
        with Image.open(T+'img/%s.jpg'%ids[k]) as im: return s.tf(im.convert('RGB'))
specs=yaml.safe_load(open('/cache/ma-user/VTBenchLab/lar/configs/models_e3.yaml'))['models']
dev=torch.device('cuda')
for i,sp in enumerate(specs):
    nm=sp['name']; out=OUT+nm.replace('/','_')+'.npy'
    if os.path.exists(out): print('skip',nm); continue
    try:
        b=MA.load_patch_bundle(sp['loader_name'],dev)
        dl=DataLoader(DS(b.eval_transform),batch_size=max(4,int(sp.get('batch_size',16))),num_workers=8)
        fs=[]
        for im in dl:
            with torch.inference_mode(), b.autocast_context():
                fs.append(b.encoder(im.to(dev,non_blocking=True)).float().cpu().numpy())
        Z=np.concatenate(fs).astype(np.float32)
        assert Z.shape[0]==len(ids) and np.isfinite(Z).all()
        np.save(out,Z); print('%3d %-32s %s'%(i,nm,Z.shape))
        del b; torch.cuda.empty_cache()
    except Exception as e:
        print('FAIL',nm,repr(e)[:140]); torch.cuda.empty_cache()
    sys.stdout.flush()
print('done')
