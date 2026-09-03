#!/usr/bin/env python
"""角度三：token 可寻址性 —— 用 LLM 自己的注意力几何，不训 projector。

到目前为止所有指标（probing / kNN / retrieval / CKA / RankMe / A,C / mutual-kNN / GW / RSA /
loss proxy / 我做的 readout battery）量的都是同一件事：**信息在不在**。
没有一个量的是：**LLM 能不能找到它**。

而 MLLM 读图的方式是注意力：文本 query 在视觉 token 上做检索。所以真正决定下游的可能不是
"信息存在于表征中"，而是"给定一个文本 query，注意力能不能把相关 token 从无关 token 里挑出来"。
这是一个几何性质（query-key 可分性），不是信息量性质，两者可以完全脱钩——一个表征可以
线性可读性极高但在 LLM 的注意力度量下完全不可寻址。

测法（全程冻结，不训练任何参数）：
  把 M+1 张图的视觉 token 拼成一条序列，后面接**其中第 0 张图的 caption**，
  跑一次前向拿 attention，统计 caption 的内容词把多少注意力质量放在第 0 张图的 token 上。
  随机命中率 = 1/(M+1)。分数 = 实际占比，跨层跨头平均。

projector 的处理：默认用随机线性投影并对多次抽样求平均。理由是随机投影近似保距（JL），
所以量到的是 encoder token 云的内在几何，而不是某次 projector 训练的成败——后者正是
gated loss proxy 的死因（pe_core / dinov2_base / pixio 三个 run 直接退化）。
也可以 --projector-root 用已训好的 projector 做对照。

预期的判决点：
  1. pe_lang_g14_448 > pe_core_g14_448（MLLM 相差 +6.15，probing 相差 −13.6）
  2. dino/mae 这些"probing 高但 MLLM 低"的族应当在这里显著偏低
若两条都不成立，这条线也关掉。
"""
import argparse, json, os, sys, re, numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lar'))
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
REPO = Path(__file__).resolve().parents[1]
QWEN = "/cache/ma-user/hf_cache/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"
STOP = set("a an the of and or with is are was were be for from by this that there it its on in at to".split())


def patch_keep_tokens():
    import model_adapters as MA
    orig = MA.PatchMeanEncoder._finish
    def patched(self, tokens):
        self.last_tokens = tokens.detach().float()
        return orig(self, tokens)
    MA.PatchMeanEncoder._finish = patched
    return MA


class ImgSet(Dataset):
    def __init__(self, paths, tf): self.p, self.tf = paths, tf
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        with Image.open(self.p[i]) as im: return self.tf(im.convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-images", type=int, default=256)
    ap.add_argument("--tokens-per-image", type=int, default=64, help="每张图下采样到这么多 token，跨 encoder 固定")
    ap.add_argument("--distractors", type=int, default=7, help="M，随机命中率 = 1/(M+1)")
    ap.add_argument("--draws", type=int, default=3, help="抽样次数（重抽干扰项与槽位）")
    ap.add_argument("--queries", type=int, default=96, help="每次抽样评测多少条 query；效应量很小，"
                    "query 数比 draw 数更重要")
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 14, 20])
    ap.add_argument("--projector-root", default="", help="用已训 projector 替代随机投影，例如 "
                    "gradient_compatibility/artifacts/full_sweep_v1/projectors")
    ap.add_argument("--out", default="mechanistic/out/addressability.json")
    args = ap.parse_args()

    MA = patch_keep_tokens()
    from data import image_path
    ids = [l.strip() for l in open(REPO / "lar/features/dinov2_large__coco4618.ids.txt")][: args.n_images]
    allid = [l.strip() for l in open(REPO / "lar/text/caption__coco4618.ids.txt")]
    caps = json.load(open(REPO / "lar/text/caption__coco4618.texts.json"))
    cmap = dict(zip(allid, caps)); caps = [cmap[i] for i in ids]
    paths = [str(image_path("coco4618", i)) for i in ids]

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(QWEN)
    llm = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.bfloat16,
                                               attn_implementation="eager").to(dev).eval()
    for p in llm.parameters(): p.requires_grad_(False)
    emb = llm.get_input_embeddings(); D = llm.config.hidden_size
    specs = {r["name"]: r for r in yaml.safe_load(open(REPO / "lar/configs/models_e3.yaml"))["models"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res = {}
    K, M = args.tokens_per_image, args.distractors
    chance = 1.0 / (M + 1)
    for nm in args.models:
        b = MA.load_patch_bundle(specs[nm]["loader_name"], dev)
        dl = DataLoader(ImgSet(paths, b.eval_transform), batch_size=max(2, int(specs[nm].get("batch_size", 8))), num_workers=6)
        Zs = []
        for imgs in dl:
            with torch.inference_mode(), b.autocast_context():
                b.encoder(imgs.to(dev))
            T = b.encoder.last_tokens
            idx = torch.linspace(0, T.shape[1] - 1, K).long().to(T.device)
            Zs.append(T[:, idx].cpu())
        Z = torch.cat(Zs); del b; torch.cuda.empty_cache()
        I, _, d_enc = Z.shape
        proj = None
        if args.projector_root:
            pp = REPO / args.projector_root / nm / "seed_0" / "projector_seen_4096.pt"
            if pp.exists():
                import torch.nn as nn
                ck = torch.load(pp, map_location="cpu")
                sd = ck["projector"] if isinstance(ck, dict) and "projector" in ck else ck
                proj = nn.Sequential(nn.Linear(d_enc, D), nn.GELU(), nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
                proj.load_state_dict({k.split("layers.")[-1] if "layers." in k else k: v
                                      for k, v in sd.items()}, strict=True)
                proj = proj.to(dev, torch.bfloat16).eval()
                for q in proj.parameters(): q.requires_grad_(False)
                print(f"  用已训 projector {pp.name}", flush=True)
            else:
                print(f"  !! 缺 {pp}，回退随机投影", flush=True)
        scores = []
        for draw in range(args.draws):
            g = torch.Generator().manual_seed(7000 + draw)
            R = (torch.randn(d_enc, D, generator=g) / np.sqrt(d_enc)).to(dev, torch.bfloat16)
            perm = torch.randperm(I, generator=g).tolist()
            got, got_mm = [], []
            for qi in perm[: min(I, args.queries)]:
                pool = [qi] + [p for p in perm if p != qi][:M]
                # 因果 LM + RoPE 有强近因偏置：固定把 own image 放在序列最前面会系统性压低它的
                # 注意力份额（实测低于随机命中率）。把 own image 的槽位随机化，让位置偏置在多次查询上抵消。
                slot = int(torch.randint(0, M + 1, (1,), generator=g))
                pool = pool[1:]; pool.insert(slot, qi)
                zb = Z[pool].to(dev, torch.bfloat16)
                with torch.inference_mode():
                    V = (proj(zb) if proj is not None else zb @ R).reshape(1, (M + 1) * K, D)
                # 配对对照：同一张图、同一组干扰项、同一个槽位，只换 caption 的内容。
                # 任何与内容无关的几何/位置偏置在差值里被抵消。
                mm = perm[(perm.index(qi) + 1 + M) % len(perm)]        # 一条与本图无关的 caption
                shares = {}
                for tag, src in (("match", qi), ("mismatch", mm)):
                    words = [w for w in re.findall(r"[a-z]+", caps[src].lower())
                             if len(w) > 2 and w not in STOP]
                    if not words:
                        shares[tag] = None; continue
                    t = tok(" " + " ".join(words), return_tensors="pt").to(dev)
                    inp = torch.cat([V, emb(t.input_ids)], 1)
                    with torch.inference_mode():
                        out = llm(inputs_embeds=inp, output_attentions=True)
                    nv = (M + 1) * K
                    per_layer = []
                    for L in args.layers:
                        A = out.attentions[L][0, :, nv:, :nv]          # [heads, text, visual]
                        mass = A.sum(-1, keepdim=True).clamp(min=1e-6)
                        own = A[:, :, slot * K:(slot + 1) * K].sum(-1, keepdim=True) / mass
                        per_layer.append(float(own.mean()))
                    shares[tag] = float(np.mean(per_layer))
                    del out, inp
                del V
                if shares.get("match") is None or shares.get("mismatch") is None: continue
                got.append(shares["match"]); got_mm.append(shares["mismatch"])
            torch.cuda.empty_cache()
            # 每条 query 自成一个配对（同图、同干扰项、同槽位，只换 caption），
            # 所以配对差的标准误直接给出效应量的误差棒。|t| < 2 就是噪声。
            d = np.array(got) - np.array(got_mm)
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
            scores.append(dict(match=float(np.mean(got)), mismatch=float(np.mean(got_mm)),
                               delta=float(d.mean()), se=se, n_query=int(len(d)),
                               t=float(d.mean() / se) if se and se > 0 else 0.0))
            print(f"  [{nm}] draw{draw} delta={d.mean():+.5f} +- {se:.5f}  t={d.mean()/se:+.2f}  "
                  f"n_q={len(d)}  (match={np.mean(got):.4f} mismatch={np.mean(got_mm):.4f})", flush=True)
        res[nm] = dict(own_share=float(np.mean([s["match"] for s in scores])),
                       mismatch=float(np.mean([s["mismatch"] for s in scores])),
                       delta=float(np.mean([s["delta"] for s in scores])),
                       se=float(np.sqrt(np.mean([s["se"] ** 2 for s in scores]) / len(scores))),
                       t=float(np.mean([s["delta"] for s in scores]) /
                               np.sqrt(np.mean([s["se"] ** 2 for s in scores]) / len(scores))),
                       per_draw=scores, n_tokens_sampled=K, d=d_enc)
        json.dump(res, open(args.out, "w"), indent=1)
        del Z; torch.cuda.empty_cache()
    print("\n=== token 可寻址性：delta = 匹配 caption − 错配 caption（主指标）===")
    for k, v in sorted(res.items(), key=lambda x: -x[1]["delta"]):
        print(f"  {k:<28} delta={v['delta']:+.5f} +- {v.get('se', float('nan')):.5f}  "
              f"t={v.get('t', 0):+.2f}   match={v['own_share']:.4f}")


if __name__ == "__main__":
    main()
