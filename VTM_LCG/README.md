# VTM-LCG

这是 [`VTM_LCG_MLLM_PREDICTOR.md`](../VTM_LCG_MLLM_PREDICTOR.md) 的独立
Phase 0 工程目录。目前实现两个结构一致的连续 CLIP tokenizer 的完整 patch-token
提取、可恢复缓存和统计检查；masked predictor、VTM/LCG 与下游排名拟合留到后续阶段。

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

## 后续命名空间

`models/`、`train/`、`eval/` 和 `rank/` 已预留给 Phase 1 及后续闭环。本阶段不提供
占位 predictor，避免误把未实现接口当成可运行协议。

