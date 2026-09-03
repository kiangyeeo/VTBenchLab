#!/usr/bin/env python
"""角度二：高范数 sink token 的因果干预。

观察（我在 21 个 encoder 上量过 token 级几何）：
  pe_lang_g14_448 是全表 MLLM 最好的模型（58.03），ImageNet probing 却只有 74.61。
  它有 0.8% 的 token 范数 > 该图中位数的 5 倍，这不到 1% 的 token 贡献了池化向量 13.6% 的模长；
  每张图 1024 个 token 的有效秩只有 4.2。其余 20 个 encoder 基本没有这个现象。

机制假设：高范数 token 是 attention sink / register。
  - 对 MLLM 有用甚至必要：给注意力一个倾倒质量的去处，让承载内容的 token 保持干净。
  - 对任何**池化**读出是灾难：池化向量被这几个 token 主导，量到的是 sink 不是图。
这一条假设同时解释了 probing 为什么在 pe_lang 上错得最厉害，也解释了为什么所有便宜 baseline
（probing / kNN / retrieval / CKA / RankMe / A,C score / mutual-kNN / GW / RSA）都在同一个地方瞎——
它们打分的对象都是一张图一个池化向量。

本脚本做的是廉价的一半（无需训练）：
  条件 A  全部 token 池化        —— 现状
  条件 B  去掉 sink 后池化       —— 预测：pe_lang 的 probing 大幅上升
  条件 C  只用 sink 池化         —— 预测：接近随机
读出用 COCO val2017 80 类多标签 mAP（比 ImageNet probing 更贴下游，且同一批图上能闭式解）。

贵的一半（你们跑，2 次训练）：把 pe_lang 的 sink token 从送进 MLLM 的序列里丢掉，重训。
  预测：MLLM 表现下降。
若 B 与这一条同时成立，就拿到了一个因果结论：sink 伤池化探针、帮 MLLM。
"""
import argparse, json, os, sys, numpy as np, torch, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lar'))
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
REPO = Path(__file__).resolve().parents[1]


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


def coco_labels(ids):
    inst = json.load(open(REPO / "data/gvt/raw/coco/annotations/instances_val2017.json"))
    cats = sorted({c["id"] for c in inst["categories"]}); cm = {c: i for i, c in enumerate(cats)}
    pos = {int(v): i for i, v in enumerate(ids)}
    Y = np.zeros((len(ids), len(cats)), np.float32)
    for a in inst["annotations"]:
        if a["image_id"] in pos: Y[pos[a["image_id"]], cm[a["category_id"]]] = 1
    return Y


def ap_mean(Yt, P):
    o = []
    for j in range(Yt.shape[1]):
        y = Yt[:, j]
        if y.sum() < 3: continue
        s = np.argsort(-P[:, j]); ys = y[s]; tp = np.cumsum(ys)
        o.append(((tp / np.arange(1, len(ys) + 1)) * ys).sum() / y.sum())
    return float(np.mean(o)) if o else float("nan")


def probe(Z, Y, seed=0):
    N = len(Z); rng = np.random.default_rng(seed); p = rng.permutation(N)
    a, b = int(.55 * N), int(.75 * N)
    Z = Z - Z.mean(0); Z = Z / (Z.std(0) + 1e-8)
    tr, va, te = p[:a], p[a:b], p[b:]
    U, S, Vt = np.linalg.svd(Z[tr], full_matrices=False); UtY = U.T @ Y[tr]
    best = (-9, None)
    for al in np.logspace(-1, 5, 10):
        W = Vt.T @ ((S / (S ** 2 + al))[:, None] * UtY)
        m = ap_mean(Y[va], Z[va] @ W)
        if m > best[0]: best = (m, W)
    return ap_mean(Y[te], Z[te] @ best[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-images", type=int, default=2000)
    ap.add_argument("--sink-mult", type=float, default=5.0, help="norm > 该倍数 * 该图中位数 -> sink")
    ap.add_argument("--topk-frac", type=float, default=0.01, help="备用定义：范数最大的这一比例 token")
    ap.add_argument("--out", default="mechanistic/out/sink.json")
    args = ap.parse_args()
    MA = patch_keep_tokens()
    from data import image_path
    ids = [l.strip() for l in open(REPO / "lar/features/dinov2_large__coco4618.ids.txt")][: args.n_images]
    paths = [str(image_path("coco4618", i)) for i in ids]
    Y = coco_labels(ids)
    specs = {r["name"]: r for r in yaml.safe_load(open(REPO / "lar/configs/models_e3.yaml"))["models"]}
    dev = torch.device("cuda"); os.makedirs(os.path.dirname(args.out), exist_ok=True)
    res = {}
    for nm in args.models:
        b = MA.load_patch_bundle(specs[nm]["loader_name"], dev)
        dl = DataLoader(ImgSet(paths, b.eval_transform), batch_size=max(4, int(specs[nm].get("batch_size", 16))), num_workers=6)
        A, B, C, frac, ratio = [], [], [], [], []
        for imgs in dl:
            with torch.inference_mode(), b.autocast_context():
                b.encoder(imgs.to(dev))
            T = b.encoder.last_tokens                                    # [B,N,D]
            n = T.norm(dim=-1)                                           # [B,N]
            med = n.median(1, keepdim=True).values
            sink = n > args.sink_mult * med
            k = max(1, int(args.topk_frac * T.shape[1]))
            topk = torch.zeros_like(sink)
            topk.scatter_(1, n.topk(k, 1).indices, True)
            sink = sink | topk                                           # 两种定义取并集
            keep = ~sink
            A.append(T.mean(1).cpu())
            kb = keep.float().unsqueeze(-1)
            B.append(((T * kb).sum(1) / kb.sum(1).clamp(min=1)).cpu())
            sb = sink.float().unsqueeze(-1)
            C.append(((T * sb).sum(1) / sb.sum(1).clamp(min=1)).cpu())
            frac.append(float(sink.float().mean()))
            ratio.append(float((n.max(1).values / (med.squeeze(1) + 1e-9)).mean()))
        del b; torch.cuda.empty_cache()
        Za, Zb, Zc = [np.concatenate([x.numpy() for x in L]).astype(np.float64) for L in (A, B, C)]
        r = dict(sink_frac=float(np.mean(frac)), norm_max_over_med=float(np.mean(ratio)),
                 mAP_all=probe(Za, Y), mAP_no_sink=probe(Zb, Y), mAP_sink_only=probe(Zc, Y))
        r["gain_from_removing_sinks"] = r["mAP_no_sink"] - r["mAP_all"]
        res[nm] = r
        print("%-28s sink=%.3f%%  max/med=%5.2f | mAP all=%.3f  去sink=%.3f (%+.3f)  只用sink=%.3f"
              % (nm, 100 * r["sink_frac"], r["norm_max_over_med"], r["mAP_all"], r["mAP_no_sink"],
                 r["gain_from_removing_sinks"], r["mAP_sink_only"]), flush=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
