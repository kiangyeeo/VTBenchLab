# VTM-LCG

这是 [`VTM_LCG_MLLM_PREDICTOR.md`](../VTM_LCG_MLLM_PREDICTOR.md) 的独立工程目录。
目前已经实现 Phase 0 特征缓存和 Phase 1 masked predictor smoke test。下游 MLLM
ground truth 与 tokenizer 排名拟合留到后续阶段。

## Phase 0 范围

- OpenAI CLIP ViT-L/14 与 MetaCLIP ViT-L/14 2.5B；
- 224×224 输入、16×16 patch 网格、256 个 1024 维 patch token；
- COCO Karpathy train 中稳定排序的前 1000 张图；
- FP16 SafeTensors 分片缓存；
- per-channel mean/std、空间 token variance、NaN/Inf 和标准化回读检查。

现有模型与数据只会被读取。项目产生的缓存和报告全部写到 `artifacts/`。

## 环境

仓库现有的 `dino` Conda 环境已包含本阶段依赖。无需安装或下载新内容：

```bash
conda activate dino
cd /cache/ma-user/VTBenchLab/VTM_LCG
```

## 运行

先运行不加载大模型的单元测试：

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

运行完整 1K smoke test：

```bash
PYTHONPATH=. python -m vtm_lcg.cache.extract \
  --config configs/phase0_smoke.yaml \
  --all
```

也可以使用统一脚本：

```bash
scripts/run_phase0_smoke.sh
```

如果当前空闲显存不足，可在不改变缓存协议的前提下降低前向 batch：

```bash
scripts/run_phase0_smoke.sh --batch-size 16
```

单独运行一个 tokenizer 或较小子集：

```bash
PYTHONPATH=. python -m vtm_lcg.cache.extract \
  --config configs/phase0_smoke.yaml \
  --tokenizer clip_openai__l14 \
  --limit 1
```

缓存位于 `artifacts/phase0_smoke/cache/<tokenizer_id>/<cache_key>/`。每个缓存目录包含：

- `identity.json`：完整缓存身份与 provenance；
- `records.json`：有序 image/caption 清单；
- `manifest.json`：分片、checksum 和完成状态；
- `shards/*.safetensors`：`values` 与 `image_ids`；
- `stats.json`：完整 mean/std 和验收指标。

两个 tokenizer 完成后，根输出目录会生成 `summary.json` 和 `summary.md`。

## Phase 1：Masked Predictor

Phase 1 使用 Phase 0 的 1000 张原始 token cache，并固定划分：

- train：800 张，只用这个 split 拟合 per-channel mean/std；
- validation：100 张，选择最佳 epoch；
- test：100 张，只计算最终 VTM、LCG 和 sanity checks。

模型配置：

```text
normalized visual tokens [B,256,1024]
  → visual projection 1024→256
  → 50% positions replaced by learned MASK token
  → fixed 2D sin/cos position embedding
  → optional frozen CLIP text tokens [B,77,768] → projection 768→256
  → concatenate visual/text sequences
  → 4-layer Transformer, d=256, 8 heads, MLP ratio=4
  → predict masked visual tokens in the normalized 1024-d space
```

同一个 predictor 使用 50% caption dropout，同时学习：

```text
visible visual tokens → masked visual tokens
visible visual tokens + caption → masked visual tokens
```

共享文本条件来自本地 OpenAI CLIP text tower。全部 5002 条 caption embedding 只编码一次，
两个视觉 tokenizer 使用完全相同的文本 cache。

### 启动

完整双 tokenizer、seed 0、20 epochs：

```bash
scripts/run_phase1_smoke.sh
```

只运行训练入口：

```bash
PYTHONPATH=. python -m vtm_lcg.train.train_predictor \
  --config configs/predictor/phase1_smoke.yaml \
  --all
```

单独训练一个 tokenizer：

```bash
PYTHONPATH=. python -m vtm_lcg.train.train_predictor \
  --config configs/predictor/phase1_smoke.yaml \
  --tokenizer clip_openai__l14
```

调试时可缩短 epoch 或降低 batch：

```bash
scripts/run_phase1_smoke.sh --epochs 2 --batch-size 16
```

相同协议重复运行时会校验 text cache 和 predictor checkpoint，完整结果不会重复训练。

### 指标

测试集上使用固定 mask，所有损失都只在 masked positions 计算：

\[
L_{\mathrm{mean}}
=
\frac{1}{|\mathcal M|D}
\sum_{t\in\mathcal M}\|\widetilde z_t\|_2^2
\]

\[
L_{\mathrm{visual}}
=
\frac{1}{|\mathcal M|D}
\sum_{t\in\mathcal M}
\|\widetilde z_t-\widehat z_t(\widetilde Z_{\bar{\mathcal M}})\|_2^2
\]

\[
L_{\mathrm{visual|text}}
=
\frac{1}{|\mathcal M|D}
\sum_{t\in\mathcal M}
\|\widetilde z_t-\widehat z_t(\widetilde Z_{\bar{\mathcal M}},C)\|_2^2
\]

\[
\mathrm{VTM}
=
1-\frac{L_{\mathrm{visual}}}{L_{\mathrm{mean}}},
\qquad
\mathrm{LCG}
=
\frac{L_{\mathrm{visual}}-L_{\mathrm{visual|text}}}
{L_{\mathrm{visual}}}.
\]

同时报告：

- shuffled caption：正确 caption 是否优于其他图片的 caption；
- spatial shuffle：只打乱 visible-token value 与原坐标的对应关系；
- no-visible：隐藏全部视觉值，只保留 mask token 和位置编码，应接近 mean baseline。

输出位于：

```text
artifacts/phase1_smoke/
├── split.json
├── text_cache/
├── predictors/<tokenizer_id>/seed_0/<run_key>/
│   ├── run_identity.json
│   ├── normalization.safetensors
│   ├── history.json
│   ├── predictor.safetensors
│   └── result.json
├── summary.json
└── summary.md
```

### 当前 smoke 结果

当前 1K/seed-0 结果中，两个 tokenizer 的 VTM 均为正，spatial shuffle 均使损失上升，
no-visible loss 约等于 mean baseline，说明视觉建模路径和空间对照工作正常。

LCG 只有 \(10^{-5}\) 量级；OpenAI CLIP 的 true-caption 未优于 shuffled-caption。
因此当前只能说明 Phase 1 工程闭环跑通，不能声称 caption 提供了稳定、特异的视觉 token
预测增益。下一步应优先增加 predictor 数据或训练 seeds，再决定是否扩大候选集合。

## 后续

`rank/` 预留给 Phase 2 之后的多 tokenizer seed stability、受控 MLLM ground truth 和
held-out tokenizer 排名拟合。

`models/`、`train/`、`eval/` 和 `rank/` 已预留给 Phase 1 及后续闭环。本阶段不提供
占位 predictor，避免误把未实现接口当成可运行协议。
