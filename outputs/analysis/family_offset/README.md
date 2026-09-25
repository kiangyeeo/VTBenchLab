# Family offset analysis (2026-09-02)

诊断 ImageNet linear probing 预测 MLLM 下游表现的失效机制。所有脚本在 `TokBench` conda env
下运行，路径写死为绝对路径，直接 `python aN_*.py` 即可。

数据来源：`lar/configs/e3_targets.csv` (n=84)、`result/VisualTokenizer表现 - MLLM详细结果 (1).csv`
(n=47 逐任务)、`lar/features/*__coco4618.npy` (84 encoder × 4618 图 patch-mean)、
`lar/text/caption__coco4618.npy` (Qwen2.5-1.5B)。

## 脚本

| 脚本 | 做什么 | 产物 |
|---|---|---|
| `a1_diag.py`  | 家族内/跨家族 Spearman、残差的家族方差分解 | `master.csv` |
| `a2_task.py`  | 11 个下游任务对 MLLM Avg 方差的贡献 | stdout |
| `a3_probe.py` | 缓存特征上的语言可读性（raw/std ridge、RBF KRR、mutual-kNN、text-CKA、谱统计） | `struct.csv` |
| `a4_pool.py`  | 覆盖率混淆：各 baseline 在自有池 vs 公共池；LLM 依赖 | stdout |
| `a5_bands.py` | probing 等值窗口内的 MLLM 跨度、受控对照对、逆序对统计 | stdout |
| `a6_struct.py`| 结构量 vs 残差；leave-one-family-out 组合 | `merged.csv` |
| `a7_coco.py`  | **标签空间消融**：COCO 主导目标 / 80 类多标签 / 小目标 / caption 词表 | `coco_probes.csv` |
| `a8_tok.py`   | token 级几何（spatial share、token 有效秩、norm 离群、token 间余弦） | `tok_stats.jsonl` |
| `a9_offset.py`| 截距 vs 斜率模型比较；偏移的 LLM 不变性；A score 解剖 | stdout |
| `a11_regimes.py` | 跨家族 / 家族内 / 强候选区间；top-1 regret | stdout |
| `a12_budget.py`  | 语言读出的样本复杂度曲线（n=100…2400） | `budget.csv` |
| `a14_sink.py` | 离群 token 对 pooled 向量的贡献（pe_lang_g14_448 案例） | stdout |
| `a15_coco.py` | 标签空间消融的完整评测（三协议 + regret + 前 25% 区间） | `a15_coco.log` |

## 主要结论

1. 失效是**每个家族一个加性截距**：`probe` 单独 R²=0.628，加家族截距 0.915，再加家族斜率只多 0.019；
   残差方差被家族身份解释 0.806。偏移跨度 −11.8 (dino) → +9.9 (pe_lang)，而 target 全域只有 31.1。
2. DINOv1 (斜率 +0.087) 与 WebSSL-MAE (+0.147) 两族里 probing 完全无信号。
3. `e3_report.md` 里的 baseline 对比有覆盖率混淆：公共池 (n=42) 上 probing 是 0.919。
   retrieval/CKA/pretrain loss 只覆盖 2/20 个 SSL；A score 覆盖 0/5 个 discrete。
4. 真正的对手是 A score（跨家族 0.909），但它在 siglip2 族内只有 0.353（probing 0.874），
   在 GT 前 25% 只有 0.441。强候选区间目前没有可用指标。
5. **已排除**：谱分散度、BN/尺度敏感度 (Waste)、非线性 headroom（符号相反）、
   对齐样本复杂度曲线形状、视觉–文本 CKA。
6. **有效**：换 probe 的**标签空间**（同特征同探针）。一族抽一个 ρ 0.360 → 0.696，
   跨族 regret 4.93 → 1.88，GT 前 25% 0.216 → 0.812。
   对照组（COCO 单一主导目标）只到 0.515 / 5.15 —— 所以是**标签结构**而非数据集分布的问题。

## 第二阶段：Readout Battery（2026-09-03）

在同一批缓存 patch-mean 特征上把"换标签空间"做成协议。每个 encoder 只做一次 SVD，被所有
标签空间复用；84 个 encoder 约 20 分钟。新增脚本 `b1`–`b3`（COCO 场景电池）和 `c1`–`c4`
（TextVQA 文字读出）。

TextVQA 图像与 OCR 标签导出在 scratchpad，重跑用 `c1_export.py`（需 dino env 的 pyarrow：
`/home/ma-user/miniconda3/envs/dino/bin/python`），特征抽取用 `c2_extract.py`（不改仓库代码）。

| score | n | 全表 ρ | 一族抽一个 | 跨族 regret | GT 前 25% |
|---|---:|---:|---:|---:|---:|
| ImageNet probing | 79 | 0.731 | 0.361 | 4.88 | 0.216 |
| gated loss proxy | 79 | 0.422 | 0.280 | 4.14 | 0.008 |
| VTB-4 场景版 (obj80+small+crowd+动作词) | 79 | 0.842 | 0.681 | 1.84 | **0.749** |
| VTB-5 全量版 (+ocr_word) | 74 | 0.872 | 0.764 | 1.24 | 0.672 |
| VTB-6 (+probing) | 74 | **0.877** | **0.765** | **1.19** | 0.570 |
| A score | 68 | 0.932 | 0.908 | 0.62 | 0.503 |

关键点：

1. 全部为等权 z-score，**无拟合**，单个新 tokenizer 可独立打分。
2. **做子集选择反而更差**（嵌套选择 0.605 / 前 25% 0.147）——固定协议是对的。
3. `oi_class` 对照成立：同一批 TextVQA 图，物体标签只到 0.525/0.535，文字标签到
   0.877/0.784，增益来自"会读字"而非图像域变化。
4. `ocr_word` 全局最强但 GT 前 25% 是 −0.125；场景版在前 25% 最好。**两个能力轴，分开汇报。**
5. 家族偏移大幅收缩：dino −11.84 → −0.99，dinov2 −5.27 → −1.18，pe_lang +9.94 → +0.95；
   残差被家族解释 0.806 → 0.657。剩余大头：raev2 −10.0、dinov3 −9.0、eupe −5.8。
6. 覆盖：场景版 79/79；全量版 74/79（vilau、uniar 的 encoder 在 TextVQA 抽取时加载失败）。
