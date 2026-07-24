# VTM-LCG

这是 [`VTM_LCG_MLLM_PREDICTOR.md`](../VTM_LCG_MLLM_PREDICTOR.md) 的独立工程目录。
目前已经实现原始 VTM-LCG 协议，以及避免全局 token 广播获得虚高分的
CV-RVTM（Cross-View Residual VTM）协议。下游 MLLM ground truth 与 tokenizer
排名拟合留到后续阶段。

## Phase 0 范围

- OpenAI CLIP ViT-L/14、MetaCLIP ViT-L/14 2.5B、SigLIP2 L/16 256 与
  MetaCLIP 2 L/14 224；
- 模型原生 224/256 输入、16×16 patch 网格、256 个 1024 维 patch token；
- COCO Karpathy train 中稳定排序的前 1000 张图；
- FP16 SafeTensors 分片缓存；
- per-channel mean/std、空间 token variance、NaN/Inf 和标准化回读检查。

现有模型与数据只会被读取。项目产生的缓存和报告全部写到 `artifacts/`。

## 环境

仓库现有的 `dino` Conda 环境已包含本阶段依赖。无需安装或下载新内容。
必须使用这个环境或其他满足 `transformers<5` 的环境；base 环境中的
Transformers 5.x 与当前 PyTorch 不兼容。

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

四个 tokenizer 完成后，根输出目录会生成 `summary.json` 和 `summary.md`。

四个 tokenizer 的共同 predictor 表面都是 `16×16×1024`，但预处理保持模型原生：

- OpenAI CLIP、MetaCLIP 1/2：224×224、CLIP mean/std；
- SigLIP2：resize 284 后 center-crop 256、mean/std 均为 0.5。

tokenizer 级 preprocess override 会进入 cache identity，不能用 224 输入替代
SigLIP2 的原生 256 输入。

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

## 完整 COCO Karpathy

完整协议直接使用官方 Karpathy 三个 split：

- train：82,783 张；
- validation：5,000 张；
- test：5,000 张。

Train、validation 和 test 分别建立 Phase 0 cache。Phase 1 只使用 train cache 的
per-channel mean/std，validation 选择最佳 epoch，test 只计算最终指标。

完整视觉 cache 的理论大小约为：

- train：40.42 GiB/tokenizer；
- validation：2.44 GiB/tokenizer；
- test：2.44 GiB/tokenizer；
- 四个 tokenizer 合计约 181.22 GiB，建议至少预留 220 GiB 空间。

完整 Phase 1 不会把全部视觉特征载入内存，而是按 Phase 0 SafeTensors shard 流式训练。
Caption 也不再预计算约 55 GiB 的全量 embedding cache，而是在每个 batch 中由冻结 CLIP
text tower 在线编码；text tower 不参与反向传播。

### 分步启动

第一步，提取三个 split 的完整视觉特征：

```bash
scripts/run_phase0_coco_karpathy_full.sh
```

如显存不足：

```bash
scripts/run_phase0_coco_karpathy_full.sh --batch-size 16
```

第二步，训练完整 COCO predictor：

```bash
scripts/run_phase1_coco_karpathy_full.sh
```

只训练一个 tokenizer：

```bash
PYTHONPATH=. python -m vtm_lcg.train.train_full_coco \
  --config configs/coco_karpathy_full/phase1_predictor.yaml \
  --tokenizer clip_openai__l14
```

### 一键启动

```bash
scripts/run_coco_karpathy_full.sh
```

Phase 0 会逐 shard 原子写入并断点续跑。完整 Phase 1 每个 epoch 保存模型、optimizer、
scheduler、最佳 checkpoint 和历史，进程中断后使用相同命令会从下一个 epoch 继续。
训练完成后，相同协议会直接校验并复用最终 checkpoint。

输出位置：

```text
artifacts/coco_karpathy_full/
├── phase0/
│   ├── train/
│   ├── validation/
│   └── test/
└── phase1/
    ├── predictors/
    ├── summary.json
    └── summary.md
```

## CV-RVTM

原始 VTM 的 MSE explained variance 会奖励跨 patch 的全局复制。CV-RVTM v1
改为对齐双视图上的位置特有残差预测：

1. view A 使用模型原生 deterministic center crop；
2. view B 使用同一 crop 的水平翻转和确定性 brightness/contrast/saturation 扰动；
3. vision encoder 前向后将 view B 的 patch columns 翻回 view A 坐标，因此两个
   `16×16` token grid 严格对齐；
4. 使用 4×4 coarse grid 上的完整 block，固定遮挡 12/16 blocks，即 75% patch；
5. 只用 source visible tokens 估计图像共享成分
   \(g_V=\operatorname{mean}_{t\in V}Z_t\)，不读取 masked target；
6. predictor 双向训练 A→B 和 B→A，目标是 \(R_t=Z_t-g_V\)；
7. 最终分数为

\[
\mathrm{CVRVTM}
=
\frac{L_{\mathrm{residual,null}}-L_{\mathrm{residual,pred}}}
{L_{\mathrm{total}}}.
\]

分母保留原始 token 总能量，因此全位置复制会得到零 residual gain；独立噪声虽然
有 residual energy，但跨视图无法预测，也不会得到高分。

测试集自动运行三个必要对照：

- collapsed tokens：每张图的所有 patch 替换为图内均值；
- independent noise：两个视图加入相互独立的确定性噪声；
- spatial shuffle：只打乱 source visible-token values 与位置的对应。

三个对照的 CV-RVTM 都应低于主结果。

### CV-RVTM smoke

提取 1K COCO train paired-view cache，并对 800/100/100 split 训练四个 tokenizer：

```bash
scripts/run_cvrvtm_smoke.sh
```

也可以分开运行：

```bash
PYTHONPATH=. python -m vtm_lcg.cvrvtm.cache \
  --config configs/phase0_smoke.yaml \
  --artifact-root artifacts/cvrvtm/phase0_smoke \
  --all

PYTHONPATH=. python -m vtm_lcg.cvrvtm.train \
  --config configs/cvrvtm/phase1_smoke.yaml \
  --all
```

### CV-RVTM full COCO

```bash
scripts/run_cvrvtm_phase0_full.sh
scripts/run_cvrvtm_phase1_full.sh
```

或一键运行：

```bash
scripts/run_cvrvtm_full.sh
```

输出位于：

```text
artifacts/cvrvtm/
├── phase0_smoke/
├── phase1_smoke/
└── coco_karpathy_full/
    ├── phase0/{train,validation,test}/
    └── phase1/
```

paired-view FP16 cache 每个 tokenizer 的完整 COCO 约 90.6 GiB，四个 tokenizer
约 362.4 GiB；连同 checkpoint、临时空间和结果，建议至少预留 420 GiB。

## 后续

`rank/` 预留给 Phase 2 之后的多 tokenizer seed stability、受控 MLLM ground truth 和
held-out tokenizer 排名拟合。

`models/`、`train/`、`eval/` 和 `rank/` 已预留给 Phase 1 及后续闭环。本阶段不提供
占位 predictor，避免误把未实现接口当成可运行协议。
