#!/usr/bin/env python
"""角度一：Realizable Gradient Fraction (RGF) —— 保留梯度，换掉功能形式。

诊断：LESS 的 cosine 衡量的是"固定模型下，样本 A 对任务 B 有没有用"，比较对象是**样本**。
拿它比较**模型**时，换 encoder 会同时换掉 h 的尺度与几何，跨 encoder 的 cosine 不可比。

改法：不看方向，看**可实现比例**。冻结 LLM，把视觉 token 经随机投影塞进视觉槽，对 caption 做
teacher forcing，反传到视觉槽激活 h 上得到 G = dL/dh —— 这是"LLM 希望视觉 token 往哪个方向动"。
projector 本身就是一个线性映射 Z -> h，所以它一阶最优更新恰好是把 G 最小二乘拟合到 Z 上。于是

    RGF = 1 - ||G - Z W*||^2 / ||G||^2        (held-out rows)

= "LLM 提出的一阶需求里，有多大比例能被这组视觉特征通过线性 projector 真正兑现"。

三个性质正好修掉之前失败的原因：
  1. 它是 R^2，无量纲，跨 encoder 可比（cosine 不可比）。
  2. 用随机初始化的 projector，不需要训练 —— 不会被"我的 recipe 在这个输入上收没收敛"污染
     （这是 gated loss proxy 的死因：pe_core / dinov2_base / pixio 三个 run 直接退化）。
  3. 梯度取在**视觉槽**上，不是 LoRA-B 上，encoder 的作用被隔离出来。

对照：RGF_shuffled 用别的图的视觉 token 配同一条 caption。RGF_delta = RGF - RGF_shuffled
剥掉"generic 结构"，只留图像特异的可兑现需求。

判决标准（先在 pe pair 上跑，1 小时内出结果）：
    RGF_delta(pe_lang_g14_448) > RGF_delta(pe_core_g14_448)
pe_lang 的 MLLM 比 pe_core 高 6.15，而 ImageNet probing 低 13.6。这一对翻不过来，梯度这条线关掉。
"""
import argparse, json, os, sys, numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lar'))
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
QWEN = "/cache/ma-user/hf_cache/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"


def keep_tokens(finish_hook):
    """让 PatchMeanEncoder 顺带把整串 token 存下来（不改仓库代码）。"""
    import model_adapters as MA
    orig = MA.PatchMeanEncoder._finish
    def patched(self, tokens):
        self.last_tokens = tokens.detach().float()
        return orig(self, tokens)
    MA.PatchMeanEncoder._finish = patched
    return MA


def ridge_r2(Z, G, alphas):
    """三段划分：fit / 选 alpha / 评测，互不重叠。返回 held-out fraction of variance explained。

    行数必须多于 d_enc，否则岭回归在无正则一侧会插值出负 R^2。调用方保证
    n_images * tokens_per_image >= 3 * d_enc。
    """
    Z = Z - Z.mean(0, keepdim=True)
    Z = Z / (Z.std(0, keepdim=True) + 1e-8)      # 逐维标准化：去掉 encoder 任意的特征尺度
    G = G - G.mean(0, keepdim=True)
    n = len(Z)
    a, b = int(0.6 * n), int(0.8 * n)
    U, S, Vt = torch.linalg.svd(Z[:a], full_matrices=False)
    UtG = U.T @ G[:a]
    best = (-9.0, None)
    for al in alphas:
        W = Vt.T @ ((S / (S ** 2 + al))[:, None] * UtG)
        r2v = 1 - ((G[a:b] - Z[a:b] @ W) ** 2).sum() / ((G[a:b]) ** 2).sum()
        if r2v > best[0]:
            best = (float(r2v), W)
    W = best[1]
    return float(1 - ((G[b:] - Z[b:] @ W) ** 2).sum() / ((G[b:]) ** 2).sum())


class ImgSet(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="models_e3.yaml 里的 name")
    ap.add_argument("--n-images", type=int, default=192)
    ap.add_argument("--tokens-per-image", type=int, default=96, help="进回归的 token 行数（跨 encoder 固定，去掉 token 数混杂）")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=2, help="随机 projector 的重复次数")
    ap.add_argument("--projector-root", default="gradient_compatibility/artifacts/full_sweep_v1/projectors",
                    help="用已训好的 projector 暖启动。随机 projector 下 LLM 看到的是噪声，"
                         "它的一阶需求与 Z 几乎无关（实测 R^2≈0），必须暖启动才有信号。留空则用随机投影。")
    ap.add_argument("--out", default="mechanistic/out/rgf.json")
    args = ap.parse_args()

    MA = keep_tokens(None)
    from data import image_path

    ids = [l.strip() for l in open(REPO / "lar/features/dinov2_large__coco4618.ids.txt")][: args.n_images]
    caps = json.load(open(REPO / "lar/text/caption__coco4618.texts.json"))
    if isinstance(caps, dict):
        caps = [caps[i] for i in ids]
    else:
        allid = [l.strip() for l in open(REPO / "lar/text/caption__coco4618.ids.txt")]
        m = dict(zip(allid, caps)); caps = [m[i] for i in ids]
    paths = [str(image_path("coco4618", i)) for i in ids]

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(QWEN)
    llm = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.bfloat16).to(dev).eval()
    for p in llm.parameters():
        p.requires_grad_(False)
    emb = llm.get_input_embeddings()
    D = llm.config.hidden_size

    specs = {r["name"]: r for r in yaml.safe_load(open(REPO / "lar/configs/models_e3.yaml"))["models"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = {}
    for name in args.models:
        b = MA.load_patch_bundle(specs[name]["loader_name"], dev)
        dl = DataLoader(ImgSet(paths, b.eval_transform), batch_size=args.batch_size, num_workers=4)
        # 1) 抽 token
        Zs = []
        for imgs in dl:
            with torch.inference_mode(), b.autocast_context():
                b.encoder(imgs.to(dev))
            Zs.append(b.encoder.last_tokens.cpu())
        Z = torch.cat(Zs)                       # [I, N, d_enc]
        del b; torch.cuda.empty_cache()
        I, N, d_enc = Z.shape
        print(f"[{name}] tokens {Z.shape}", flush=True)

        proj = None
        if args.projector_root:
            pp = REPO / args.projector_root / name / "seed_0" / "projector_seen_4096.pt"
            if pp.exists():
                ck = torch.load(pp, map_location="cpu")
                sd = ck["projector"] if isinstance(ck, dict) and "projector" in ck else ck
                import torch.nn as nn
                proj = nn.Sequential(nn.Linear(d_enc, D), nn.GELU(), nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
                proj.load_state_dict({k.split("layers.")[-1] if "layers." in k else k: v for k, v in sd.items()},
                                     strict=True)
                proj = proj.to(dev, torch.bfloat16).eval()
                for q in proj.parameters():
                    q.requires_grad_(False)
                print(f"  暖启动 projector: {pp.name}", flush=True)
            else:
                print(f"  !! 找不到 {pp}，回退随机投影（预期无信号）", flush=True)

        per_seed = []
        for seed in range(args.seeds):
            g = torch.Generator().manual_seed(1000 + seed)
            R = (torch.randn(d_enc, D, generator=g) / np.sqrt(d_enc)).to(dev, torch.bfloat16)
            rows_Z, rows_G, rows_Gs = [], [], []
            perm = torch.randperm(I, generator=g)          # shuffled control: 换一张图的视觉 token
            for start in range(0, I, args.batch_size):
                sel = list(range(start, min(start + args.batch_size, I)))
                for mode in ("real", "shuf"):
                    src = sel if mode == "real" else [int(perm[j]) for j in sel]
                    zb = Z[src].to(dev, torch.bfloat16)
                    with torch.no_grad():
                        V0 = proj(zb) if proj is not None else zb @ R
                    V = V0.clone().requires_grad_(True)                                    # [B,N,D]
                    txt = tok([caps[j] for j in sel], return_tensors="pt", padding=True,
                              truncation=True, max_length=48).to(dev)
                    te = emb(txt.input_ids)
                    inp = torch.cat([V, te], 1)
                    out = llm(inputs_embeds=inp).logits[:, V.shape[1] - 1 : -1]
                    lab = txt.input_ids.clone(); lab[txt.attention_mask == 0] = -100
                    loss = torch.nn.functional.cross_entropy(
                        out.reshape(-1, out.shape[-1]).float(), lab.reshape(-1), ignore_index=-100)
                    G, = torch.autograd.grad(loss, V)
                    idx = torch.randperm(N, generator=g)[: args.tokens_per_image]
                    if mode == "real":
                        rows_Z.append(Z[sel][:, idx].reshape(-1, d_enc).double())
                        rows_G.append(G[:, idx].reshape(-1, D).float().cpu().double())
                    else:
                        rows_Gs.append(G[:, idx].reshape(-1, D).float().cpu().double())
                    del V, G, out, inp
                torch.cuda.empty_cache()
            Zr = torch.cat(rows_Z); Gr = torch.cat(rows_G); Gsr = torch.cat(rows_Gs)
            gm = Gr.mean(0, keepdim=True)
            share_global = float((gm ** 2).sum() * len(Gr) / (Gr ** 2).sum())
            print("  诊断: G 中被所有 token 共享的常向量占比 = %.3f (接近 1 = LLM 的需求与具体 token 无关)" % share_global, flush=True)
            al = [10.0 ** k for k in range(-2, 8)]
            if len(Zr) < 3 * d_enc:
                print(f"  !! 行数 {len(Zr)} < 3*d_enc {3*d_enc}: 调大 --n-images 或 --tokens-per-image", flush=True)
            r_real = ridge_r2(Zr, Gr, al)
            r_shuf = ridge_r2(Zr, Gsr, al)
            per_seed.append(dict(seed=seed, rgf=r_real, rgf_shuffled=r_shuf, rgf_delta=r_real - r_shuf,
                                 grad_norm=float(Gr.norm())))
            print(f"  seed{seed}  RGF={r_real:.4f}  shuffled={r_shuf:.4f}  delta={r_real-r_shuf:+.4f}", flush=True)
        results[name] = dict(per_seed=per_seed,
                             rgf=float(np.mean([s["rgf"] for s in per_seed])),
                             rgf_delta=float(np.mean([s["rgf_delta"] for s in per_seed])),
                             n_tokens=N, d=d_enc)
        json.dump(results, open(args.out, "w"), indent=1)
        del Z; torch.cuda.empty_cache()
    print("\n=== RGF ===")
    for k, v in sorted(results.items(), key=lambda x: -x[1]["rgf_delta"]):
        print(f"  {k:<28} RGF={v['rgf']:.4f}  delta={v['rgf_delta']:+.4f}  (N={v['n_tokens']}, d={v['d']})")


if __name__ == "__main__":
    main()
