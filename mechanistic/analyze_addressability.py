#!/usr/bin/env python
"""跑完 addressability 后算三协议。用法：
    python mechanistic/analyze_addressability.py mechanistic/out/addressability_full.json
"""
import json, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
REPO = "/cache/ma-user/VTBenchLab/"
res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else REPO + "mechanistic/out/addressability_full.json"))
t = pd.read_csv(REPO + "lar/configs/e3_targets.csv").rename(columns={"name": "tok"})
rows = [dict(tok=k, delta=v["delta"], match=v["own_share"], mismatch=v["mismatch"]) for k, v in res.items()]
M = pd.DataFrame(rows).merge(t[["tok", "family", "qwen2_5"]], on="tok").dropna(subset=["qwen2_5"])
print("n=%d  families=%d" % (len(M), M.family.nunique()))
rng = np.random.default_rng(0)
for col in ["delta", "match"]:
    fs = list(M.family.unique()); rs = []; rg = []
    for _ in range(4000):
        idx = [rng.choice(M.index[M.family == f]) for f in fs]; g = M.loc[idx]
        rs.append(spearmanr(g[col], g.qwen2_5).statistic)
    for _ in range(8000):
        k = min(5, len(fs)); pick = rng.choice(len(fs), k, replace=False)
        idx = [rng.choice(M.index[M.family == fs[j]]) for j in pick]; g = M.loc[idx]
        rg.append(g.qwen2_5.max() - g.qwen2_5.values[g[col].values.argmax()])
    print("%-8s 全表 rho=%+.3f  一族抽一个=%+.3f  跨族 regret=%.2f"
          % (col, spearmanr(M[col], M.qwen2_5).statistic, np.mean(rs), np.mean(rg)))
print("\n判决：一族抽一个 > 0.5 且 dino/webssl_mae 族显著偏低 -> 继续；否则关掉。")
print(M.groupby("family")[["delta", "qwen2_5"]].mean().sort_values("qwen2_5").round(5).to_string())
print("\n逐模型：")
print(M.sort_values("delta", ascending=False)[["tok", "family", "delta", "qwen2_5"]].round(5).to_string(index=False))
