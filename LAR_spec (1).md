# LAR (Language-Accessible Ratio) — 实现说明

## 0. 目标

给每个 visual tokenizer 算两个**训练无关**的标量,用于预测该 tokenizer 训出的 MLLM 表现。

- `LAR`:语言可用的信息中,落在特征前 m 个主方向上的比例
- `Waste`:方差中"没有语义"的部分占比

全流程**不训练任何参数**,单模型 < 10 分钟。

---

## 1. 数据

两套数据,分工不同。**所有路径已就位。**

### 1.1 主数据集 —— COCO val2017(用于 LAR,即 E2 / E3)

```
图像       /cache/ma-user/VTBenchLab/data/gvt/raw/coco/val2017                       (5,000)
caption   /cache/ma-user/VTBenchLab/data/gvt/raw/coco/annotations/captions_val2017.json  (25,014)
answer    /cache/ma-user/VTBenchLab/data/gvt/raw/vqa/v2_mscoco_val2014_annotations.json
```

**关键便利:VQAv2 val 的图就是 COCO 的图**,所以视觉特征只抽一次,两个文本 domain 共用同一份 `Z`,只换 `E`。

#### 行的单位 = 图(硬性要求)

`Z` 的每一行必须是**一张图**,绝不能是 (图, answer) 对。若把 12,949 条 answer 各占一行、视觉特征复制多份,answer 多的图会在协方差矩阵里被超额加权,**λ 谱直接失真**——而 λ 是这个指标的一半。

#### 两个 domain 的样本集

| domain | 图数 | 说明 |
|---|---|---|
| `caption` | 5,000 | 全覆盖 |
| `answer` | 4,618 | 筛 `answer_type == "other"` 后的覆盖数,382 张无 other-answer,直接丢弃 |

**主对照跑在共同的 4,618 张上**(唯一变量是文本),把图 id 落盘成 `configs/image_ids_4618.txt`。

另外**加跑一次 `caption` 在全部 5,000 张上**作为稳健性检查:两者若差异明显,说明 N=5,000 不足以稳定估谱,走 §1.2 的两阶段方案。

#### answer 的选取规则

每图取**一条**,不做平均。规则:

```python
# 该图所有 answer_type == "other" 的问题中,question_id 最小的那条
qid = min(q["question_id"] for q in other_type_questions_of(image_id))
text = annotation[qid]["multiple_choice_answer"]     # VQAv2 的 10 人众数答案
```

理由:每图 answer 数从 1 到 8 不等(平均 2.8),若做**平均**,answer 多的图 embedding 方差被平滑得更低,产生与 answer 数相关的异方差伪影。取单条则完全没有这个问题,且与杨毅霖 `a_only` (Sp 0.928) 的设定一致。

可选稳健性变体(`E` 只算一次,几乎零成本):每图最多取 3 条求 embedding 均值,并把 `answer_count` 存进输出表,检查它是否与 LAR 相关。

### 1.2 谱估计不稳时的两阶段方案(仅在 §1.1 稳健性检查不过时启用)

train2017 图像未下载,COCO 无法扩样本。但 ImageNet 是全的,可以把两个估计解耦:

```
主方向 v_k、方差 λ_k   →  在 ImageNet-1k val 50,000 张上估(大 N,谱稳定)
方向有用性 r_k         →  把 COCO 4,618 张投影到固定的 v_k 上再算(需要配对文本)
```

即 `Zk_coco = (Z_coco - mean_imagenet) @ V_imagenet`。两个估计用同一组方向,合法且干净;代价是引入 ImageNet→COCO 的轻微域偏移,所以**只作 fallback,不作主方案**。

### 1.3 E1 专用 —— ImageNet-1k val

```
/cache/ma-user/VTBenchLab/data/imagenet1k        (val 50,000 / 1,000 类)
```

E1 要验证 `Waste` 能否预测 BatchNorm 增益,而 BN gain 本身就是在 **ImageNet linear probe** 上测的,必须同分布才公平。

- 图像:val 随机 10,000 张(固定 seed,落盘 id 列表)
- 文本:类名 prompt,`"a photo of a {classname}"`,用同一个 LLM 编码

> **注意:ImageNet 不能用于 E2 / E3。** 它只有 1000 个类名、没有自由文本,`r_k` 会退化成"该方向对类别可不可分",LAR 就变回 linear probing 了。而 DINO 系(全表最大残差 −11.5)恰恰是"语义存在但非词汇化",只有自由文本才区分得出来。

### 1.4 文本编码器

Qwen2.5-1.5B(与主表 MLLM backbone 一致),取**最后一层 hidden state 的 mean-pool**。

**每个 domain 只跑一次,所有 encoder 共用同一份 `E`。**

### 1.5 通用要求

所有 encoder 用**同一批图、同一顺序**,图像 id 列表落盘。

---

## 2. 核心计算

### 2.1 视觉特征 Z

对每个 encoder `f`:

1. 前向 N 张图,取 patch token 序列,**mean-pool 成一个向量** `z_i ∈ R^d`
   - 若模型有 CLS token,统一**只用 patch mean-pool**,不用 CLS(保证全模型口径一致)
2. 拼成 `Z ∈ R^{N×d}`
3. **只做中心化,不做 L2 归一化**(我们要测的就是方差结构,归一化会把它抹掉)

```python
Z = Z - Z.mean(axis=0, keepdims=True)
```

### 2.2 谱分解

```python
U, S, Vt = np.linalg.svd(Z, full_matrices=False)   # Z: N×d
lam = S**2 / (N - 1)          # λ_k, 长度 d
Zk  = U * S                   # N×d, 第 k 列 = 第 k 主方向的投影 z_k
```

### 2.3 文本特征 E

```python
E = E - E.mean(0);  E = E / (E.std(0) + 1e-8)      # N×D_text
```

### 2.4 方向级语言有用性 r_k

$$r_k = \frac{1}{D}\sum_{j=1}^{D}\mathrm{corr}^2\!\left(z_k,\;E_{:,j}\right)$$

向量化实现:

```python
Zs = (Zk - Zk.mean(0)) / (Zk.std(0) + 1e-8)        # N×d
C  = (Zs.T @ E) / N                                # d×D, 每格是一个相关系数
r  = (C**2).mean(axis=1)                           # 长度 d
r  = np.maximum(r - 1.0/N, 0.0)                    # 有限样本偏差校正: E[corr²]≈1/N
```

### 2.5 指标

```python
w = lam * r                                        # 每个方向的"语言可用信息量"
LAR_m  = w[:m].sum() / w.sum()                     # m ∈ {8,16,32,64,128}
LAR_f  = w[:int(0.05*d)].sum() / w.sum()           # 固定比例版本(见 §2.6)

r_norm = r / (r.max() + 1e-12)
Waste  = (lam * (1 - r_norm)).sum() / lam.sum()
```

### 2.6 必须做的混杂控制

| 混杂 | 处理 |
|---|---|
| 特征维度 d 不同(384–1664) | **同时报 `LAR_m`(固定 m)和 `LAR_f`(固定 m/d)两个版本** |
| 谱尾在 N 有限时噪声大,且 d 越大尾巴越长 | **分母统一截断到前 `K = min(d, 512)` 个方向**,所有模型用同一个 K 规则 |
| token 数不同(49–729) | 不做 resize,但把 token 数存进输出表作为协变量 |
| N 有限导致 r_k 有正偏差 | 已在 §2.4 减 1/N |

---

## 3. 三个验证实验(按顺序做,前一个不过就停)

### E1 — Waste 能否预测 BatchNorm 增益(第 1 天)

- 数据:**ImageNet-1k val 10,000 张 + 类名文本**(§1.2),与 BN gain 的测量分布一致
- 输入:5–8 个模型的 `probe_with_BN` 与 `probe_without_BN`
  - 已有数据点:WebSSL MAE-300M `+15.62pp`、DINOv1 ViT-S/16 `+0.00pp`
  - 需补:再挑 4–6 个覆盖 siglip2 / metaclip / dinov2 / pixio / eupe
- 输出:`Spearman(Waste, BN_gain)`
- **通过标准:ρ > 0.7。不过则整条路停,损失一天。**

### E2 — LAR 是族级量还是族内可用(第 2 天)

```
F = between-family variance / within-family variance     # 12 个 family
```

同时输出每个 family 内部的 `Spearman(LAR, MLLM_Avg)`。

判读:F 很大 + 族内 ρ 接近 0 → LAR 只能当**档位轴**指标,必须与 probing/kNN 组合,不能单独用。

### E3 — 三协议评估(第 3 天)

target:`MLLM Avg (qwen2.5)`,以及三个 LLM 列的 PC1。

三个协议都要报:

1. **全表 Spearman**
2. **一族抽一个 Spearman** — 每个 family 随机抽 1 个模型算 ρ,重复 5,000 次取均值
3. **top-1 regret (k=5)** — 随机抽 5 个模型,指标选 top1,记 `max(GT) − GT(pick)`,重复 20,000 次取均值

还要测**组合**:`probe_epoch1 + LAR` 的 regret(留 5 个,在其余上拟合线性回归,再预测这 5 个的 top1)。

对照 baseline:`probe_epoch1`、`retrieval-IN`、`CKA`、`pretrain loss`、`A score`、`RankMe`、`eff_rank`。

---

## 4. 输出格式

`results/lar_metrics.csv`,每个 (encoder × text_domain × image_set) 一行:

```
name, text_domain, image_set, d, n_tokens, N,
lam_top128        (分号分隔的 128 个数)
r_top128          (分号分隔的 128 个数)
LAR_8, LAR_16, LAR_32, LAR_64, LAR_128, LAR_frac05,
Waste, eff_rank, RankMe
```

`image_set` 取值:`coco4618` / `coco5000` / `in1k10k`。

若跑了 answer 的多条平均变体,另存一列 `answer_count_mean` 用于检查异方差伪影。

保留完整谱是为了事后能重画 `Reach(m)` 曲线,不用重跑。

---

## 5. 代码结构

```
lar/
  extract_visual.py    # encoder -> features/{name}.npy  (N×d, float32)
  extract_text.py      # Qwen2.5 -> text/{domain}.npy    (只跑一次)
  compute_lar.py       # Z, E -> results/lar_metrics.csv
  eval_e1.py           # Waste vs BN gain
  eval_e2.py           # F 比 + 族内 rho
  eval_e3.py           # 三协议 + 组合
  configs/models.yaml  # encoder 名称、权重路径、分辨率、pooling 方式
```

---

## 6. 备注

- 所有 encoder 冻结,`torch.no_grad()`,fp16 前向即可
- `Z` 用 float32 存盘;SVD 前转 float64 避免数值问题
- 若某模型 `d > N`,SVD 只取前 N 个方向(秩上限)
- `E` 只算一次,不要每个 encoder 重算
