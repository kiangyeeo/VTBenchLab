# VTM-LCG：用视觉 Token 可建模性预测下游 MLLM 表现

> 工作定位：像 AC Score 一样，构造一套远低于完整 MLLM 训练成本的代理协议，预测不同连续视觉 tokenizer 在固定 MLLM 配方下的下游表现。

---

## 0. 一句话结论

这个 idea 可以做。

第一版只研究连续 CLIP 类视觉 tokenizer：

```text
图像
  → 冻结 CLIP vision encoder
  → 完整 patch-token sequence
  → tiny masked Transformer
      ├── 只看可见视觉 token：测视觉 token 可建模性 VTM
      └── 再看 caption：测语言条件增益 LCG
  → 用 VTM + LCG 拟合受控 MLLM 的下游表现
  → 预测未参与拟合的视觉 tokenizer 排名
```

核心假设是：

> 如果一个视觉 tokenizer 输出的 patch-token sequence 具有稳定的内部规律，并且语言能够有效解释其中与语义相关的不确定性，那么固定容量的 Transformer 应当更容易学习和使用这些 token，最终也更可能在受控 MLLM 中取得更好的下游表现。

这里的目标不是证明低预测损失本身等于“好视觉表示”，而是检验：

> **VTM 和 LCG 能否像 AC Score 中的 A、C 一样，成为下游 MLLM 表现的低成本预测变量。**

---

## 1. 研究问题

设第 \(i\) 个冻结视觉 tokenizer 为 \(E_i\)。对图像 \(x\)，它输出连续视觉 token：

\[
Z_i=E_i(x)\in\mathbb R^{N_i\times D_i}.
\]

将 \(E_i\) 接入固定的 MLLM 配方后，得到下游表现：

\[
Y_{i,d}
=
\operatorname{Score}_d
\left(
\operatorname{MLLM}(E_i)
\right),
\]

其中 \(d\) 表示 caption、VQA、OCR 或细粒度感知等能力域。

本项目希望在不为每个新 tokenizer 完整训练一次 MLLM 的情况下，用两个便宜指标预测 \(Y_{i,d}\)：

1. **Visual Token Modelability，VTM**  
   可见视觉 token 能否预测被遮挡视觉 token。

2. **Language-Conditioned Gain，LCG**  
   加入图像 caption 后，被遮挡视觉 token 的预测误差下降多少。

最终学习一个类似 AC Score 的映射：

\[
\widehat Y_{i,d}
=
f_d(\mathrm{VTM}_i,\mathrm{LCG}_i).
\]

研究是否成功，不取决于拟合数据上的 \(R^2\) 是否漂亮，而取决于它能否预测**未参与拟合的 tokenizer**。

---

## 2. 为什么这个假设合理

### 2.1 MLLM 实际处理的是视觉 token sequence

常见理解型 MLLM 的视觉路径是：

```text
image
  → vision encoder
  → patch-token sequence
  → connector/projector
  → LLM
  → text answer
```

LLM 接收到的不是一个 GAP 后的全局向量，而是一串视觉 token。因此，只测试全局分类或图文 embedding 相似度，可能遗漏：

- token 之间的局部关系；
- 空间结构；
- 重复和冗余；
- 全局语义是否分散在多个 token 中；
- Transformer 是否容易从有限数据中学会读取这些 token。

### 2.2 VTM 测量视觉序列的内部规律

如果遮住部分视觉 token，剩余 token 能够较好地预测它们，说明该表示具有可被固定容量 Transformer 利用的结构。

VTM 可能同时反映：

- 局部连续性；
- 对象内部一致性；
- 跨区域关系；
- token 顺序和二维位置结构；
- 表示中的噪声与不稳定性。

### 2.3 LCG 测量语言能解释多少视觉残差

仅仅容易预测不一定是好事。高度重复甚至近似塌缩的 token 也可能很容易预测。

因此进一步询问：

> 在已经看到部分视觉 token 的情况下，加入对应 caption 后，还能额外减少多少被遮挡视觉 token 的不确定性？

如果 caption 能显著降低预测误差，说明视觉 token 中存在与语言描述一致、且能被小型 Transformer 访问的语义结构。

LCG 与普通图文 embedding alignment 的区别是：

- alignment 通常比较全局向量距离；
- LCG 直接作用于完整视觉 token sequence；
- LCG 测量的是语言对 token-level 预测任务的实际帮助；
- 它自然包含 token 顺序、空间位置和局部—全局关系。

### 2.4 为什么它可能预测下游 MLLM

固定 MLLM 训练配方后，不同视觉 tokenizer 的主要差异来自：

1. 是否保留任务相关视觉信息；
2. 信息是否具有容易学习的结构；
3. 信息是否与语言语义兼容；
4. connector 和 LLM 需要多少训练才能读出这些信息。

VTM 主要对应第 2 项，LCG 主要对应第 3 项。

因此它们不一定单独决定 MLLM 表现，但有希望成为有效预测变量。

---

## 3. 与 AC Score 的对应关系

AC Score 的基本路线是：

```text
为视觉表示构造 A、C 两个便宜指标
  → 用少量完整 MLLM 结果拟合映射
  → 预测其他视觉表示的 MLLM 表现
```

本项目采用同样的研究范式：

```text
为视觉 tokenizer 构造 VTM、LCG 两个便宜指标
  → 用受控 MLLM ground truth 拟合映射
  → 预测未参与拟合的 tokenizer
```

对应关系如下：

| AC Score 路线 | VTM-LCG 路线 |
|---|---|
| A：跨模态 alignment | LCG：语言对视觉 token 预测的增益 |
| C：视觉 correspondence | VTM：视觉 token sequence 的内部可建模性 |
| 二次函数拟合 MLLM 表现 | 线性、交互项或二次函数拟合 MLLM 表现 |
| 用部分完整训练结果校准 | 用部分受控 MLLM 结果校准 |
| 预测未训练候选 | 预测未参与拟合的 tokenizer |

本项目不是要复制 AC 的两个具体指标，而是采用相同的“低成本指标 + 少量校准 + 未见候选预测”协议。

---

## 4. 第一阶段研究边界

为了让最初实验简单且可解释，第一阶段只做：

- 连续视觉 tokenizer；
- CLIP、OpenCLIP、MetaCLIP 等 CLIP 类 vision encoder；
- 最后一层完整 patch-token sequence；
- 固定输入分辨率；
- 固定 patch size；
- 固定 token 数；
- 尽量固定 hidden dimension；
- 单图输入；
- caption 作为语言条件；
- 随机 masked-token prediction；
- 固定 MLLM、connector、数据和训练配方。

第一阶段暂时不做：

- 离散视觉 token；
- 连续与离散统一排名；
- 不同 token 数的强行比较；
- OCR transcript、scene graph、object list 等多种条件；
- 视频；
- 动态分辨率；
- 多种复杂 mask；
- 任意公开 MLLM 的跨系统预测。

第一阶段只回答：

> 在结构相近的连续 CLIP tokenizer 中，VTM 和 LCG 能否预测固定 MLLM 配方下的相对表现？

---

## 5. Tokenizer 输入定义

### 5.1 主 representation surface

最重要的原则是：

> predictor 读取的视觉 token，必须与受控 MLLM 实际送入 connector 的视觉 token 是同一个 representation surface。

第一阶段为所有 CLIP 类视觉 tokenizer 预注册同一种 surface，例如：

```text
最后一个视觉 Transformer block
  → 模型原生的 final normalization（若该接口包含）
  → 去掉 CLS/register tokens
  → 保留全部 patch tokens
  → 不经过仅服务于全局检索的 visual projection
```

如果受控 MLLM 使用倒数第二层，那么 predictor 也必须使用倒数第二层；不能在代理实验和下游实验中选择不同层。

输出：

\[
Z_i\in\mathbb R^{N\times D}.
\]

第一阶段优先选择相同 \(N,D\) 的模型。例如：

- 输入分辨率：224；
- patch size：14；
- patch tokens：\(16\times16=256\)；
- hidden dimension：1024。

这样可以避免输入投影维度、token 数和训练 FLOPs 成为主要混淆。

### 5.2 不使用 CLS 或 mean pooling

主实验不得把视觉表示压成 \([B,D]\)。

原因是本项目要测：

- token sequence 的内部结构；
- token-level 语言可解释性；
- 局部与全局关系；
- Transformer 对完整视觉序列的学习难度。

CLS 和 mean pooling 只能作为已有静态 baseline。

### 5.3 特征标准化

不同 checkpoint 的 feature scale 可能不同。对每个 tokenizer，在训练 split 上拟合：

\[
\mu_{i,d}
=
\mathbb E[Z_{i,:,d}],
\qquad
\sigma_{i,d}
=
\sqrt{\operatorname{Var}(Z_{i,:,d})+\epsilon}.
\]

标准化 token：

\[
\widetilde Z_{i,:,d}
=
\frac{Z_{i,:,d}-\mu_{i,d}}{\sigma_{i,d}}.
\]

\(\mu_i,\sigma_i\) 只能在训练集上拟合，并冻结用于 validation/test。

标准化的目的只是消除数值尺度差异，不改变 tokenizer 的空间和语义结构。

---

## 6. Tiny Masked Transformer

### 6.1 输入与目标

对每张图像的 patch-token sequence 随机选择遮挡集合 \(\mathcal M\)，第一阶段固定：

\[
|\mathcal M|/N=50\%.
\]

模型看到：

- 未遮挡视觉 token \(\widetilde Z_{\bar{\mathcal M}}\)；
- 所有 patch 的二维位置编码；
- 被遮挡位置上的 `[MASK]` token；
- 可选 caption 条件。

模型预测：

\[
\widehat Z_{\mathcal M}.
\]

这里遮挡的是 vision encoder 已经生成的 contextualized patch tokens，而不是遮挡原始图像 patch。可见 token 可能已经通过视觉 encoder 的 self-attention 包含其他位置的信息。这不是实现错误：本项目测的是 **MLLM 实际收到的输出 token sequence 是否冗余、规则且容易读取**。

但如果 50% random mask 使任务过于简单、所有 tokenizer 的损失几乎相同，应先提高 mask ratio 或改用 block mask，而不是立即增加 predictor 容量。

### 6.2 最小模型配置

建议第一版：

```yaml
visual_input_dim: 1024
model_dim: 256
depth: 4
num_heads: 8
mlp_ratio: 4
mask_ratio: 0.5
dropout: 0.0
position_embedding: fixed_2d_sincos
loss: normalized_mse
```

这通常属于约 4M–10M 可训练参数量级，足以作为第一版 tiny predictor。

所有 tokenizer 必须使用：

- 相同结构；
- 相同初始化协议；
- 相同训练图片；
- 相同 batch 顺序；
- 相同 mask seed；
- 相同 optimizer；
- 相同训练步数；
- 相同训练 token 数；
- 相同验证集。

### 6.3 用一个模型同时得到两个损失

不必为每个 tokenizer 分别训练 unconditional 和 conditional 两个模型。

训练时以固定概率丢弃 caption，例如：

```yaml
caption_dropout: 0.5
```

同一个 predictor 同时学习：

```text
无 caption：
  visible visual tokens → masked visual tokens

有 caption：
  visible visual tokens + caption → masked visual tokens
```

验证时分别计算两种损失，减少训练成本和模型差异。

### 6.4 Caption 条件必须固定

所有视觉 tokenizer 必须使用同一个 caption 处理方式：

- 同一个文本 tokenizer；
- 同一个冻结文本 encoder；
- 同一个文本 hidden dimension；
- 同一个语言到 predictor 的 projection；
- 同一个最大文本长度。

不能让每个 CLIP 使用自己的 text encoder，否则测到的会是整套 CLIP 图文系统差异，而不只是视觉 tokenizer 差异。

最简单的融合方式是：

```text
frozen text encoder outputs
  → shared text projection
  → 与 visual tokens 拼接
  → modality/type embedding
  → masked Transformer
```

---

## 7. 指标定义

### 7.1 Mean-prediction baseline

最简单的无视觉上下文基线是预测训练集均值。

标准化后均值接近 0，定义：

\[
L_{\mathrm{mean},i}
=
\frac{1}{|\mathcal M|D}
\sum_{t\in\mathcal M}
\left\|
\widetilde z_{i,t}-0
\right\|_2^2.
\]

也可以显式使用训练集 position-wise mean，但第一阶段统一模型的空间结构后，全局 per-channel mean 已足够。

### 7.2 Visual modeling loss

无 caption 时：

\[
L_{\mathrm{visual},i}
=
\frac{1}{|\mathcal M|D}
\sum_{t\in\mathcal M}
\left\|
\widetilde z_{i,t}
-
\widehat z_{i,t}
\left(
\widetilde Z_{i,\bar{\mathcal M}}
\right)
\right\|_2^2.
\]

定义归一化 Visual Token Modeling Cost：

\[
\mathrm{VTMC}_i
=
\frac{L_{\mathrm{visual},i}}
     {L_{\mathrm{mean},i}}.
\]

越低表示相对于均值预测器，tiny Transformer 越容易从视觉上下文恢复被遮挡 token。

为了让“越高越好”，同时定义：

\[
\mathrm{VTM}_i
=
1-\mathrm{VTMC}_i
=
1-
\frac{L_{\mathrm{visual},i}}
     {L_{\mathrm{mean},i}}.
\]

主文可以报告 VTM，原始结果表必须保留三个量：

- \(L_{\mathrm{mean}}\)；
- \(L_{\mathrm{visual}}\)；
- VTM。

### 7.3 Language-conditioned modeling loss

加入 caption \(C\) 后：

\[
L_{\mathrm{visual|text},i}
=
\frac{1}{|\mathcal M|D}
\sum_{t\in\mathcal M}
\left\|
\widetilde z_{i,t}
-
\widehat z_{i,t}
\left(
\widetilde Z_{i,\bar{\mathcal M}},C
\right)
\right\|_2^2.
\]

### 7.4 Language-Conditioned Gain

定义：

\[
\mathrm{LCG}_i
=
\frac{
L_{\mathrm{visual},i}
-
L_{\mathrm{visual|text},i}
}{
\max(L_{\mathrm{visual},i},\epsilon)
}.
\]

解释：

- \(\mathrm{LCG}>0\)：caption 帮助预测视觉 token；
- \(\mathrm{LCG}\approx0\)：caption 基本没有提供额外信息；
- \(\mathrm{LCG}<0\)：caption 干扰预测，或模型训练不稳定。

LCG 是相对损失下降比例，因此比直接比较不同 tokenizer 的原始 MSE 更稳健。

### 7.5 Shuffled-caption control

将 batch 中 caption 随机打乱，计算：

\[
L_{\mathrm{visual|shuffled\ text},i}.
\]

期望：

\[
L_{\mathrm{visual|text},i}
<
L_{\mathrm{visual|shuffled\ text},i}.
\]

定义辅助语言特异性增益：

\[
\mathrm{LCG}^{\mathrm{specific}}_i
=
\frac{
L_{\mathrm{visual|shuffled\ text},i}
-
L_{\mathrm{visual|text},i}
}{
\max(L_{\mathrm{visual},i},\epsilon)
}.
\]

第一版回归仍使用简单 LCG；shuffled-caption 只作为必要 sanity check。

---

## 8. AC-style 下游预测器

### 8.1 Ground-truth MLLM 分数

为每个视觉 tokenizer 运行受控 MLLM 训练，得到：

\[
Y_{i,d}.
\]

所有候选必须固定：

- LLM family 和 checkpoint；
- connector 结构；
- tokenizer 冻结策略；
- 图文 alignment 数据；
- instruction-tuning 数据；
- 数据顺序；
- optimizer 和学习率；
- batch size；
- 训练步数；
- prompt template；
- 下游 benchmark；
- 解码和评分协议。

唯一允许变化的是视觉 tokenizer。

否则 \(Y_i\) 的变化不能归因于 tokenizer，代理指标也没有清晰预测目标。

### 8.2 第一版拟合函数

tokenizer 数较少时，先使用带交互项的简单模型：

\[
\widehat Y_{i,d}
=
\beta_{0,d}
+
\beta_{1,d}\mathrm{VTM}_i
+
\beta_{2,d}\mathrm{LCG}_i
+
\beta_{3,d}
\mathrm{VTM}_i\mathrm{LCG}_i.
\]

它只有四个参数，比完整二次函数更适合小样本。

当 tokenizer 设置达到约 12–15 个以上时，再测试 AC-style 二次函数：

\[
\widehat Y_{i,d}
=
\beta_{0,d}
+
\beta_{1,d}\mathrm{VTM}_i
+
\beta_{2,d}\mathrm{LCG}_i
+
\beta_{3,d}\mathrm{VTM}_i\mathrm{LCG}_i
+
\beta_{4,d}\mathrm{VTM}_i^2
+
\beta_{5,d}\mathrm{LCG}_i^2.
\]

为了降低小样本过拟合，拟合时建议使用 ridge regression，并固定正则强度选择协议。

### 8.3 预测目标

第一版分别预测：

- COCO caption；
- VQAv2；
- 两者标准化后的简单平均。

不建议一开始混入大量能力差异很大的 benchmark。

如果 VTM-LCG 对 caption 有效、对 VQA 无效，这本身就是有意义的结果，说明指标更接近语言描述一致性，而不是通用视觉推理。

---

## 9. 最小实验闭环

### 9.1 Smoke-test 候选

优先选择相同架构和输入规格的连续 tokenizer，例如：

1. OpenAI CLIP ViT-L/14；
2. MetaCLIP ViT-L/14 400M；
3. MetaCLIP ViT-L/14 2.5B；
4. 一个额外 OpenCLIP ViT-L/14 checkpoint。

四个设置只能验证：

- 工程能否跑通；
- VTM、LCG 是否有非平凡差异；
- loss 和排名是否对 seed 稳定；
- caption 是否真的带来条件增益。

四个设置不能支持可靠的“排名预测规律”结论。

### 9.2 正式 pilot 候选

建议：

```text
8–12 个 tokenizer settings
至少 4 个不同预训练来源
尽量统一 ViT-L/14、224、256 tokens、1024 dims
```

可以把不同 checkpoint 视为 tokenizer setting，但正式验证应避免所有模型都来自同一个预训练 family。

### 9.3 数据

第一版使用 COCO Karpathy split：

- train：训练 masked predictor；
- validation：选择训练步数和检查收敛；
- test：计算最终 VTM、LCG；
- 每张图存在多个 caption 时，训练阶段随机采样一个，测试阶段固定或对多个 caption 平均。

predictor 训练数据与完整 MLLM 的训练数据可以重叠，因为代理目标本来就是测这些视觉 token 在图文分布上的可学习性。

但是：

- predictor test split 不得用于训练；
- MLLM 下游 test benchmark 不得参与拟合代理模型；
- 所有 tokenizer 必须看到完全相同的数据实例。

### 9.4 Predictor 训练

建议初始配置：

```yaml
epochs: 20
global_batch_size: 256
optimizer: adamw
learning_rate: 3.0e-4
weight_decay: 0.05
warmup_ratio: 0.05
precision: bf16
caption_dropout: 0.5
mask_ratio: 0.5
seeds: [0, 1, 2]
```

最终应按相同的 image exposure 或训练 FLOPs 比较，而不只按 epoch 数比较。

### 9.5 Controlled MLLM

第一版可以沿用现有 GVT/LLaVA 风格训练：

```text
frozen vision tokenizer
  → fixed connector
  → fixed LLM
  → alignment
  → instruction tuning
  → COCO caption / VQAv2 evaluation
```

为了让 ground truth 可信：

- 每个 tokenizer 至少 2 个训练 seed；
- 对性能差异小于 seed noise 的模型视为 tie；
- tokenizer 排名应基于 seed mean；
- 报告 ground-truth 排名本身的稳定性。

---

## 10. 验证协议

### 10.1 不使用训练集拟合优度作为主证据

在少量 tokenizer 上使用二次函数，很容易得到很高的 in-sample \(R^2\)。

因此主结果必须来自未见 tokenizer：

1. leave-one-tokenizer-out；
2. tokenizer 足够多后使用 leave-one-family-out；
3. 最终保留 2–3 个完全不参与设计的 hidden test tokenizer。

### 10.2 主评价指标

报告：

- Spearman \(\rho\)；
- Kendall \(\tau_b\)；
- pairwise ranking accuracy；
- top-1/top-3 recall；
- mean absolute prediction error；
- bootstrap confidence interval。

其中最重要的是：

- 排名相关性；
- 选错最佳 tokenizer 的 regret；
- 未见 family 上的泛化。

### 10.3 必要 baselines

至少比较：

1. random ranking；
2. ImageNet zero-shot accuracy；
3. linear probe；
4. 原始 masked loss \(L_{\mathrm{visual}}\)；
5. 只用 VTM；
6. 只用 LCG；
7. VTM + LCG；
8. AC 中的 alignment/A-score 类指标，如果能够在同一候选上计算。

核心问题不是 VTM-LCG 是否与下游相关，而是：

> VTM + LCG 是否比现有便宜指标更准确地预测未见 tokenizer 的 MLLM 排名？

### 10.4 建议的 go/no-go 标准

继续扩大项目的建议标准：

- predictor 的 3 个 seed 下 VTM/LCG 排名基本稳定；
- true-caption LCG 显著高于 shuffled-caption control；
- held-out Spearman \(\rho\) 约为 0.6 或更高；
- pairwise ranking 明显好于 random；
- VTM + LCG 优于只用 zero-shot、linear probe、VTM 或 LCG；
- 至少在 caption 和 VQA 中有一个能力域成立。

如果只在同一 checkpoint family 内成立，应将结论收缩为：

> VTM-LCG 可以预测同类 CLIP tokenizer 的相对退化或提升。

如果对未见 family 完全失效，就不应继续宣传为通用 MLLM tokenizer predictor。

---

## 11. 最重要的 sanity checks

第一版只需以下检查：

### 11.1 Caption shuffle

正确 caption 应比随机 caption 更能降低 masked prediction loss。

如果随机 caption 也产生相同增益，说明模型可能只是利用：

- 文本长度；
- 数据集类别先验；
- 额外参数路径；
- 训练分布偏差。

### 11.2 Spatial-token shuffle

只在评测时打乱 visible-token values 对应的空间坐标，同时保持 mask、二维位置编码和 predictor 权重不变。

预期 VTM 应下降。否则 predictor 可能主要做 bag-of-token 统计，没有利用空间结构。

如果 token values 和二维坐标一起做同一个置换，结果理论上可以不变，因此那不是有效的空间结构对照。

### 11.3 No-visible-token baseline

让模型只看到 mask token 和位置编码。

它应接近 mean/position-wise baseline。否则数据中可能存在位置泄漏。

### 11.4 Seed stability

同一 tokenizer 的 predictor seed 方差不应大于 tokenizer 之间的主要差异。

如果 seed noise 很大，应先增加训练数据或训练预算，而不是继续拟合下游排名。

### 11.5 Information-retention baseline

CLIP 类 encoder 通常不存在离散 codebook collapse，但仍应保留：

- zero-shot；
- linear probe；
- token variance；
- effective rank。

它们用于确认极低预测损失不是由低方差表示造成的。

第一阶段不需要把这些指标合成复杂总分。

---

## 12. 可能的结果与解释

### 12.1 VTM 和 LCG 都有效

如果二者组合明显优于单项和已有 baseline，最强结论是：

> 连续视觉 token 的内部可建模性与语言条件可解释性共同预测受控 MLLM 的下游表现。

### 12.2 只有 LCG 有效

说明 CLIP token 的内部可预测性差异可能不是关键，真正重要的是视觉 token 与语言语义的兼容性。

此时项目可收缩为 Language-Conditioned Visual Token Gain。

### 12.3 只有 VTM 有效

说明视觉 token 的规律性或噪声水平比 caption alignment 更能解释下游差异。

需要进一步检查 VTM 是否只是与：

- encoder scale；
- patch variance；
- zero-shot accuracy；
- effective rank

重复。

### 12.4 两者只预测 caption，不预测 VQA

说明协议主要测图文描述一致性，不足以覆盖：

- 计数；
- OCR；
- 空间关系；
- 视觉推理。

后续可分别加入 question-conditioned 或 region-mask track，但不应在第一版提前增加。

### 12.5 两者都无效

可能原因包括：

- CLIP tokenizer 间的主要差异不在 token 可建模性；
- tiny predictor 的归纳偏置与 MLLM 不一致；
- connector compatibility 才是决定因素；
- 完整 MLLM 训练覆盖了 tokenizer 的初始差异；
- ground-truth MLLM 排名本身不稳定；
- 候选 tokenizer 数量过少或差异范围太窄。

如果经过 seed、容量和 ground-truth 检查后仍无效，就应停止这个方向，而不是继续增加复杂指标。

---

## 13. 后续扩展顺序

只有第一阶段成立后，再按以下顺序扩展。

### 13.1 不同 hidden dimension

加入固定随机投影：

\[
P_i:\mathbb R^{D_i}\rightarrow\mathbb R^{256}.
\]

使用多个固定 projection seed，检查排名稳定性。

### 13.2 不同 token 数

增加两条轨道：

1. native-token：保留原生 token 数；
2. fixed-\(K\)：统一压缩到固定视觉 token 数。

同时报告 predictor 的实际 FLOPs。

### 13.3 更困难的 mask

依次加入：

- block mask；
- quadrant mask；
- object-region mask。

这些 track 可区分局部插值和全局结构恢复能力。

### 13.4 其他语言条件

caption 成立后，再测试：

- OCR transcript；
- object list；
- scene graph；
- question；
- question + answer。

每种条件应作为独立能力轨，而不是直接平均成一个总分。

### 13.5 离散 tokenizer

离散 token 使用 cross-entropy/bits-per-token，不与连续 MSE 直接混排。

先分别验证连续、离散 track 的预测能力，再比较它们对相同下游 MLLM 目标的排名相关性。

---

## 14. 在 VTBenchLab 中的实现建议

### 14.1 不修改现有 pooled linear-probe 输出

现有 `scripts/linear_probe_tokenizers/feature_extractors.py` 主要返回 pooled \([B,D]\) 表示。

本项目应新增 sequence extractor，不应改变已有 linear-probe baseline 的语义。

### 14.2 统一 TokenBatch

建议复用并落实已有 CVLC 文档中的接口思想：

```python
@dataclass
class TokenBatch:
    values: Tensor              # [B, N, D]
    mask: BoolTensor            # [B, N]
    coords: Tensor              # [B, N, 2]
    special_mask: BoolTensor    # [B, N]
    grid_shape: tuple[int, int]
    surface: str
    input_size: tuple[int, int]
    metadata: dict
```

第一阶段连续 CLIP 不需要 `ids`。

### 14.3 建议目录

```text
vtm_lcg/
├── adapters/
│   ├── base.py
│   ├── openai_clip.py
│   ├── openclip.py
│   └── metaclip.py
├── cache/
│   ├── extract.py
│   └── dataset.py
├── models/
│   ├── masked_predictor.py
│   └── text_conditioner.py
├── train/
│   └── train_predictor.py
├── eval/
│   ├── compute_scores.py
│   └── sanity_checks.py
├── rank/
│   ├── fit_score.py
│   └── evaluate_holdout.py
└── schemas.py

configs/vtm_lcg/
├── tokenizers/
├── predictor/
└── mlmm_ground_truth/
```

### 14.4 Cache 内容

每个缓存必须记录：

```yaml
tokenizer_id:
checkpoint_sha256:
source_commit:
input_resolution:
patch_size:
token_count:
hidden_dim:
representation_surface:
normalization:
image_id:
caption_ids:
feature_dtype:
```

缓存 key 必须包含 checkpoint、预处理、surface 和代码版本，避免错误复用。

### 14.5 结果表

每个 tokenizer、predictor seed 和 split 保存：

```text
tokenizer_id
family
predictor_seed
L_mean
L_visual
L_visual_text
L_visual_shuffled_text
VTM
LCG
LCG_specific
zero_shot
linear_probe
mlmm_caption_score
mlmm_vqa_score
```

拟合程序只读取标准结果表，不直接依赖 tokenizer 推理代码。

---

## 15. 第一阶段执行顺序

### Phase 0：接口和数据检查

1. 为 2 个 CLIP tokenizer 返回完整 \([B,N,D]\) patch tokens；
2. 确认 token 数、hidden dimension 和 surface；
3. 在 1K COCO 图像上建立缓存；
4. 检查 per-channel mean/std、token variance 和 NaN。

### Phase 1：Predictor smoke test

1. 训练一个 4-layer masked Transformer；
2. 检查 train/validation loss；
3. 计算 VTM；
4. 加入 caption dropout；
5. 计算 LCG；
6. 运行 shuffled-caption 和 spatial-token shuffle。

### Phase 2：四 tokenizer 工程实验

1. 对 4 个相同结构 CLIP tokenizer 完整提取 COCO token；
2. 每个 tokenizer 训练 3 个 predictor seed；
3. 比较 VTM、LCG 的方差和排序；
4. 确认指标不是数值尺度造成的。

### Phase 3：受控 MLLM 小闭环

1. 为这 4 个 tokenizer 运行相同 MLLM 配方；
2. 得到 caption/VQA ground truth；
3. 只做散点图和简单相关性；
4. 不在 4 个点上宣称可靠回归规律。

### Phase 4：正式预测实验

1. 扩展到 8–12 个 tokenizer setting；
2. 锁定指标和训练协议；
3. 预留 hidden tokenizer；
4. 拟合简单交互模型；
5. 运行 leave-one-tokenizer-out；
6. tokenizer 足够多后运行 leave-one-family-out；
7. 与 zero-shot、linear probe 和 A-score baseline 比较。

---

## 16. 项目的成功叙事

如果结果成立，最稳妥的论文主张是：

> We introduce a low-cost protocol that characterizes continuous visual tokenizers through token-sequence modelability and language-conditioned predictive gain, and show that these factors predict the downstream ranking of controlled MLLMs on unseen visual-tokenizer settings.

中文表述：

> 我们提出一种低成本协议，从视觉 token sequence 的可建模性和语言条件预测增益两个维度描述连续视觉 tokenizer，并证明这些指标能够预测未见 tokenizer 在受控 MLLM 中的下游排名。

不应声称：

- 低 masked loss 普遍等价于好视觉 tokenizer；
- 一个分数可以预测任意 MLLM；
- 已经统一解决连续和离散 tokenizer 评估；
- 在拟合数据上获得高 \(R^2\) 就证明规律成立。

---

## 17. 最终最小版本

如果只保留最核心实验，完整协议可以缩成：

```text
候选：
  8–12 个结构相近的 CLIP vision encoders

数据：
  COCO image-caption

视觉输入：
  final normalized patch tokens，不使用 CLS/GAP

代理模型：
  同一个 4-layer、d=256 masked Transformer
  mask 50% patch tokens
  caption dropout 50%

指标：
  VTM = 1 - L_visual / L_mean
  LCG = (L_visual - L_visual|text) / L_visual

下游 ground truth：
  固定 LLM、connector、训练数据和训练步骤
  只替换 vision encoder
  评测 COCO caption 和 VQAv2

拟合：
  Y_hat = β0 + β1 VTM + β2 LCG + β3 VTM×LCG

验证：
  leave-one-tokenizer-out
  最终 hidden tokenizer
  Spearman / Kendall / pairwise ranking

对照：
  zero-shot、linear probe、VTM-only、LCG-only、A-score
```

这就是第一版应执行的完整 idea。只有这个闭环成立后，才值得增加不同分辨率、不同 token 数、复杂 mask、OCR 条件和离散 tokenizer。

---

## 18. 相关工作

- [AudioCodecBench](https://arxiv.org/abs/2509.02349)：使用 reconstruction、ID stability、Transformer perplexity 和 downstream probe 评估音频 codec。
- [GigaTok](https://arxiv.org/abs/2504.08736)：使用小型 AR Probing 模型的 validation loss、gFID 和 linear accuracy 评估视觉 tokenizer 对下游 AR 生成的影响。
- [Law of Vision Representation in MLLMs](https://arxiv.org/abs/2408.16357)：使用 alignment 和 correspondence 构造 AC Score，并拟合受控 MLLM 表现。
- [What Makes for Good Visual Tokenizers for Large Language Models?](https://arxiv.org/abs/2305.12223)：研究不同视觉预训练方式对 MLLM 语义和细粒度感知的影响。
- [Conditional Probing](https://aclanthology.org/2021.emnlp-main.122/)：从 conditional usable information 角度测量加入某个表示后，相对 baseline 增加了多少可预测信息。
