# DINO / DINOv2 ImageNet-1k k-NN Benchmark — Reproduced vs Official Gold

k-NN classification on ImageNet-1k (frozen features, cosine similarity, temperature 0.07),
评测协议统一用 DINOv2 的 `dinov2/eval/knn.py`(`nb_knn = [10, 20, 100, 200]`,eval transform = resize 256 → center-crop 224 → ImageNet 归一化)。

**主指标:20-NN top-1**(DINOv2 官方报的就是 20-NN;DINO 官方 model zoo 的 k-NN 列与之同协议)。
单元格规则:`复现值 (官方金标准值)`,保留两位小数。

Sources:
- 复现值:`outputs/<run>/results_eval_knn.json`(每个模型的 `('full', 20) Top 1`)。
- DINOv2 金标准:官方仓库 README「ImageNet k-NN」列(facebookresearch/dinov2)。
- DINO 金标准:官方仓库 model zoo「k-NN」列(facebookresearch/dino)。

## 汇总(20-NN top-1, %)

| 方法 | 骨干 | 参数量 | 复现 20-NN top-1 | 官方金标准 | 复现 (金标准) | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| DINOv2 | ViT-S/14 | 21 M | 79.13 | 79.0 | 79.13 (79.0) | +0.13 |
| DINOv2 | ViT-B/14 | 86 M | 82.11 | 82.1 | 82.11 (82.1) | +0.01 |
| DINOv2 | ViT-L/14 | 300 M | 83.50 | 83.5 | 83.50 (83.5) | 0.00 |
| DINOv2 | ViT-g/14 | 1100 M | 83.54 | 83.5 | 83.54 (83.5) | +0.04 |
| DINO | ViT-S/8 | 21 M | 78.33 | 78.3 | 78.33 (78.3) | +0.03 |
| DINO | ViT-B/16 | 86 M | 75.88 | 76.1 | 75.88 (76.1) | −0.22 |
| DINO | ViT-B/8 | 86 M | 77.28 | 77.4 | 77.28 (77.4) | −0.12 |

所有 7 个模型复现值与官方金标准的偏差均在 **±0.22% 以内**,k-NN 复现成立。

> 备注:DINO 两个 ViT-B 变体在 **k=10** 时更贴近官方(B/16: 76.11,B/8: 77.33),官方表未注明所用 k;统一取 20-NN 便于横向对比,这点小差属正常。

## 附:各模型完整 k 扫描(top-1 / top-5, %)

| 方法 | 骨干 | k=10 | k=20 | k=100 | k=200 |
| --- | --- | --- | --- | --- | --- |
| DINOv2 | ViT-S/14 | 78.99 / 91.63 | 79.13 / 92.97 | 77.89 / 94.30 | 77.17 / 94.41 |
| DINOv2 | ViT-B/14 | 82.04 / 92.75 | 82.11 / 93.92 | 81.41 / 95.24 | 80.88 / 95.47 |
| DINOv2 | ViT-L/14 | 83.54 / 93.21 | 83.50 / 94.34 | 82.90 / 95.65 | 82.54 / 95.97 |
| DINOv2 | ViT-g/14 | 83.51 / 92.90 | 83.54 / 93.99 | 82.49 / 95.22 | 81.90 / 95.49 |
| DINO | ViT-S/8 | 78.33 / 90.90 | 78.33 / 92.40 | 76.92 / 93.66 | 76.10 / 93.79 |
| DINO | ViT-B/16 | 76.11 / 89.76 | 75.88 / 91.18 | 74.28 / 92.72 | 73.20 / 92.67 |
| DINO | ViT-B/8 | 77.33 / 90.53 | 77.28 / 92.05 | 75.81 / 93.51 | 74.96 / 93.61 |
