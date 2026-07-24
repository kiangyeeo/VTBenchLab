# 基于 Transformer 兼容性的 Visual Tokenizer 下游性能预测方案

## 1. 项目目标

本项目希望在**不训练完整 MLLM、不引入语言模型**的情况下，以较低成本预测不同 visual tokenizer 接入下游 MLLM 后的表现。

核心假设：

> 如果 visual tokenizer 输出的 tokens 能被统一的小型 Transformer 快速读取、条件化查询并进行关系组合，那么它通常也更容易被下游 MLLM 的 connector 和 Transformer 模块利用。

项目测量五类能力：

1. **全局可读性**：场景、类别和属性是否容易读出；
2. **条件可寻址性**：给定查询后能否定位相关视觉信息；
3. **关系可组合性**：能否组合多个 tokens 完成关系、比较和计数；
4. **局部与细节保持**：小物体、位置、文字等是否仍可读取；
5. **Token 效率**：在有限 token budget 下能力能否保持。

---

## 2. 统一实验框架

对 tokenizer \(T_i\) 的参数完全冻结：

\[
Z_i=T_i(x)\in\mathbb{R}^{N_i\times d_i}
\]

统一评测链路为：

\[
Z_i\rightarrow P_i\rightarrow F_\theta\rightarrow H_c
\]

其中：

- \(P_i\)：参数量受控的输入 projector；
- \(F_\theta\)：统一的小型 Transformer；
- \(H_c\)：任务对应的分类、回归或匹配头；
- 不使用 LLM；
- 不做自然语言生成；
- 只训练 projector、Transformer 和任务头。

### 推荐初始配置

| 模块 | 配置 |
|---|---|
| Transformer | 2 layers |
| Hidden size | 512 |
| Attention heads | 8 |
| FFN ratio | 4 |
| Query tokens | 8 或 16 |
| Optimizer | AdamW |
| Training budgets | 500、2,000、8,000 steps |
| Seeds | 3 |
| Token budgets | 16、32、64、128 |
| Frozen | Visual tokenizer |
| Trainable | Projector、Transformer、task head |

### Projector 公平性

不同 tokenizer 的输出维度 \(d_i\) 不同。建议使用低秩 projector：

\[
P_i(z)=W_{2,i}\sigma(W_{1,i}z)
\]

其中：

\[
W_{1,i}\in\mathbb{R}^{d_i\times r_i},\qquad
W_{2,i}\in\mathbb{R}^{r_i\times d}
\]

根据 \(d_i\) 调整 \(r_i\)，使不同 tokenizer 的 projector 参数量近似一致。

---

## 3. 任务与数据集

## 3.1 全局语义读出

### 目的

测量视觉 tokens 中的全局语义是否容易被 Transformer 读取。

### 数据集与任务

| 数据集 | 任务 | 指标 |
|---|---|---|
| ImageNet-1k | 单标签分类 | Top-1 Accuracy |
| MSCOCO | 多标签物体识别 | mAP |
| Visual Genome | 物体、属性分类 | Macro Accuracy / mAP |

### 模型

在视觉 tokens 前加入 CLS token：

\[
H=F_\theta([\mathrm{CLS};Z_i])
\]

使用 \(H_{\mathrm{CLS}}\) 完成分类。

该任务主要作为基础语义控制项，不能单独代表 Transformer 兼容性。

---

## 3.2 条件化查询读出

### 目的

测量 Transformer 能否根据查询，从 visual tokens 中选择相关信息。

### 数据集

使用 **GQA Balanced** 的 scene graph 和 functional program。

### 查询形式

不输入自然语言，而是将问题转换为结构化 query，例如：

```text
operation = query_attribute
object = car
attribute = color
```

编码为：

\[
q=e_{\mathrm{operation}}+e_{\mathrm{object}}+e_{\mathrm{attribute}}
\]

输入模型：

\[
H=F_\theta([q;Z_i])
\]

### 子任务

| 类型 | 内容 |
|---|---|
| Object | 识别指定对象 |
| Attribute | 查询颜色、材质、大小 |
| Existence | 判断指定对象是否存在 |
| Relation | 查询空间或语义关系 |
| Comparison | 比较两个对象的属性 |
| Counting | 统计满足条件的对象数量 |

### 指标

- 各类型 Accuracy；
- Macro Average Accuracy；
- Query AULC；
- Query Benefit。

Query Benefit：

\[
G_{i,c}^{\mathrm{query}}
=S_{i,c}^{\mathrm{query}}-S_{i,c}^{\mathrm{no-query}}
\]

差值必须和绝对性能一起报告。

---

## 3.3 组合与关系推理

### 目的

测量 visual tokens 是否支持多步筛选、关系查询、比较和计数。

### 数据集

使用 **CLEVR** 及其 functional programs。

示例：

```text
filter_color(red)
filter_shape(cube)
relate(left)
count
```

将操作序列编码为：

\[
Q=[q_1,q_2,\ldots,q_m]
\]

输入：

\[
H=F_\theta([Q;Z_i])
\]

### 分组方式

按 program length：

| 难度 | Program length |
|---|---:|
| 简单 | 1 step |
| 中等 | 2 steps |
| 较难 | 3 steps |
| 复杂 | 4 steps 及以上 |

按推理类型：

- Attribute query；
- Spatial relation；
- Counting；
- Comparison；
- Logical composition。

### 指标

组合保持率：

\[
R_i^{\mathrm{comp}}
=
\frac{S_i^{4+}-S_{\mathrm{chance}}}
{S_i^1-S_{\mathrm{chance}}}
\]

组合性能下降：

\[
D_i^{\mathrm{comp}}=S_i^1-S_i^{4+}
\]

同时报告各 program length 的 Accuracy 和 AULC。

---

## 3.4 局部可寻址性

### 目的

测量某个对象或区域能否从 visual token 序列中被单独定位和读取。

### 数据集

使用 **Visual Genome**：

- Object boxes；
- Object labels；
- Attributes；
- Relationships。

### 子任务

1. 给定对象类别预测 bounding box；
2. 给定对象查询属性；
3. 从候选区域中选择目标区域；
4. Phrase-to-region matching；
5. 给定两个对象预测关系。

### 指标

- Bounding-box mIoU；
- Acc@IoU 0.5；
- Region Recall@1 / Recall@5；
- Attribute Accuracy；
- Relation Accuracy；
- 小、中、大物体分组结果。

---

## 3.5 OCR 与高频细节读取

### 目的

测量 tokenizer 是否保留文字、数字、小目标和高频局部细节。

### 数据集

第一阶段优先使用可控数据：

- 自行生成的字符、数字和短词；
- SynthText；
- TextOCR 的封闭答案子集。

### 子任务

1. 单字符识别；
2. 数字识别；
3. 固定长度字符串识别；
4. 查询第 \(k\) 个字符；
5. 查询指定区域中的文字；
6. 候选词选择。

### 指标

- Character Accuracy；
- Exact Match；
- Normalized Edit Distance；
- Position-conditioned Accuracy；
- 不同字号、文本长度分组结果。

归一化编辑距离：

\[
\operatorname{NED}
=1-
\frac{\operatorname{EditDistance}(\hat y,y)}
{\max(|\hat y|,|y|)}
\]

---

## 4. 对照模型

每项任务至少训练三种 readout。

### 4.1 GAP + Linear

\[
\bar z=\frac{1}{N}\sum_{n=1}^{N}z_n
\]

\[
\hat y=W\bar z+b
\]

测量简单的全局线性可读性。

### 4.2 GAP + MLP

\[
\hat y=\operatorname{MLP}(\bar z)
\]

用于控制非线性映射能力和模型参数量。

### 4.3 Token Transformer

\[
\hat y=H_c(F_\theta(Z))
\]

或：

\[
\hat y=H_c(F_\theta([Q;Z]))
\]

定义 Transformer Gain：

\[
G_i^{\mathrm{TF}}
=S_i^{\mathrm{Transformer}}-S_i^{\mathrm{MLP}}
\]

必须同时报告：

\[
\left(S_i^{\mathrm{Transformer}},G_i^{\mathrm{TF}}\right)
\]

避免低 MLP 基线导致虚假高增益。

---

## 5. 核心指标

## 5.1 Final Performance

在最大训练预算 \(B\) 下：

\[
S_{i,c}^{\mathrm{final}}=S_{i,c}(B)
\]

## 5.2 Learning Curve AUC

训练预算：

\[
b\in\{0,500,2000,8000\}
\]

记录验证集性能 \(S_{i,c}(b)\)，在对数训练步数轴上计算：

\[
\operatorname{AULC}_{i,c}
=
\frac{
\sum_j
\frac{S_{i,c}(b_j)+S_{i,c}(b_{j+1})}{2}
\left[
\log(b_{j+1}+1)-\log(b_j+1)
\right]
}{
\log(B+1)
}
\]

AULC 是主要指标，综合反映学习速度、少样本可用性和最终性能。

## 5.3 Token-budget Efficiency

统一 token budget：

\[
k\in\{16,32,64,128\}
\]

记录性能 \(S_{i,c}(k)\)，计算：

\[
E_{i,c}^{\mathrm{token}}
=\operatorname{AUC}_{\log k}S_{i,c}(k)
\]

32-token 保持率：

\[
R_{i,c}^{32}
=
\frac{S_{i,c}(32)-S_{\mathrm{chance}}}
{S_{i,c}(N_i)-S_{\mathrm{chance}}}
\]

## 5.4 Query Benefit

\[
G_{i,c}^{\mathrm{query}}
=S_{i,c}^{\mathrm{query}}-S_{i,c}^{\mathrm{no-query}}
\]

## 5.5 Composition Retention

\[
R_i^{\mathrm{comp}}
=
\frac{S_i^{4+}-S_{\mathrm{chance}}}
{S_i^1-S_{\mathrm{chance}}}
\]

## 5.6 Seed Stability

\[
V_{i,c}=\operatorname{Std}_s S_{i,c,s}
\]

兼容性好的 tokenizer 应同时具备高性能、高 AULC 和低方差。

## 5.7 达到阈值的训练成本

\[
B_i(\tau)=\min\{b:S_i(b)\geq\tau\}
\]

用于衡量 tokenizer 达到相同能力水平需要多少训练预算。

---

## 6. 最终能力向量

每个 tokenizer 输出：

\[
\Phi_i=
[
A_i^{\mathrm{global}},
A_i^{\mathrm{query}},
A_i^{\mathrm{relation}},
A_i^{\mathrm{local}},
A_i^{\mathrm{OCR}},
E_i^{\mathrm{token}},
R_i^{\mathrm{comp}},
V_i
]
\]

含义：

| 指标 | 含义 |
|---|---|
| \(A^{\mathrm{global}}\) | 全局语义 AULC |
| \(A^{\mathrm{query}}\) | 条件查询 AULC |
| \(A^{\mathrm{relation}}\) | 关系与组合 AULC |
| \(A^{\mathrm{local}}\) | 局部定位和属性读取 |
| \(A^{\mathrm{OCR}}\) | 文字与高频细节读取 |
| \(E^{\mathrm{token}}\) | Token-budget efficiency |
| \(R^{\mathrm{comp}}\) | 复杂推理保持率 |
| \(V\) | 跨 seed 方差 |

第一阶段不学习复杂总分，先分析各维度与不同下游任务的关系。

---

## 7. 下游 Gold Standard

使用已有完整 MLLM 结果作为真实预测目标：

- COCO Caption；
- VQAv2；
- Object Counting；
- Multi-class Identification；
- 其他已完成的 MLLM 任务。

每个任务内部标准化：

\[
\widetilde Y_{i,t}
=
\frac{Y_{i,t}-\mu_t}{\sigma_t}
\]

整体分数：

\[
Y_i^{\mathrm{overall}}
=
\frac{1}{T}\sum_{t=1}^{T}\widetilde Y_{i,t}
\]

整体结果与单任务结果必须同时报告。

### Proxy 与下游任务的预期对应

| 下游任务 | 主要 Proxy |
|---|---|
| COCO Caption | Global AULC、Attribute、Relation |
| VQAv2 | Query AULC、Global、Local |
| Object Counting | Composition、Counting、Token Efficiency |
| MCI | Global、Local、Query |
| OCR 类任务 | OCR、Local、Low-budget Retention |

---

## 8. 预测有效性验证

## 8.1 排名相关性

主要计算：

- Spearman \(\rho\)；
- Kendall \(\tau_b\)；
- Pearson \(r\) 仅作辅助。

## 8.2 Pairwise Ranking Accuracy

\[
A_{\mathrm{pair}}
=
\frac{1}{|\mathcal P|}
\sum_{(i,j)\in\mathcal P}
\mathbf 1
[
\operatorname{sign}(\hat Y_i-\hat Y_j)
=
\operatorname{sign}(Y_i-Y_j)
]
\]

## 8.3 Top-1 Regret

\[
i^*=\arg\max_i\hat Y_i
\]

\[
\operatorname{Regret}=Y_{\mathrm{best}}-Y_{i^*}
\]

## 8.4 增量解释能力

基础模型：

\[
Y_i
=
\alpha
+\beta_1S_i^{\mathrm{linear}}
+\beta_2S_i^{\mathrm{reconstruction}}
+\epsilon_i
\]

加入 Transformer compatibility：

\[
Y_i
=
\alpha
+\beta_1S_i^{\mathrm{linear}}
+\beta_2S_i^{\mathrm{reconstruction}}
+\beta_3S_i^{\mathrm{TF}}
+\epsilon_i
\]

需要比较的 baseline：

- Linear probing；
- k-NN；
- TokBench；
- Reconstruction metrics；
- Retrieval；
- Token 数；
- 输入分辨率；
- Tokenizer 参数量。

项目成立要求：加入 Transformer compatibility 后，相关性、pairwise accuracy 或 leave-family-out 预测明显改善。

---

## 9. 数据划分与公平性

## 9.1 图像去重

- GQA、Visual Genome 和 COCO 存在来源重叠；
- 不使用下游测试图像训练 proxy；
- 对 gold test 图像建立 blacklist；
- 检查 proxy train/val/test 与下游数据的图像 ID 重叠。

## 9.2 按 Tokenizer Family 验证

同一 tokenizer 的不同：

- checkpoint；
- 分辨率；
- token 数；
- 压缩率；
- 模型规模；

不能被随机拆到预测模型的训练集和测试集，应使用 leave-one-family-out。

## 9.3 三条输入协议

### Native-resolution Track

使用 tokenizer 实际接入 MLLM 时的推荐分辨率，作为主要结果。

### Common-resolution Track

使用所有 tokenizer 均支持的公共分辨率，例如 256，控制分辨率因素。

### Matched-token Track

统一输出 16、32、64、128 个 tokens，比较压缩效率。

---

## 10. 实验链路

### 阶段一：建立 Gold Standard

1. 汇总 tokenizer × 下游任务矩阵；
2. 检查指标方向和量纲；
3. 计算任务内排名；
4. 标准化并计算 Overall；
5. 分析下游任务间 rank correlation。

### 阶段二：统一 Tokenizer 接口

统一输出：

```python
{
    "tokens": tokens,              # [B, N, D]
    "attention_mask": mask,        # [B, N]
    "spatial_shape": (H, W),       # optional
    "native_resolution": size,
    "token_type": "continuous"    # or discrete
}
```

### 阶段三：统一 Projector

实现：

```text
[B, N_i, d_i] -> [B, N_i, 512]
```

控制 projector 参数量、初始化和归一化方式。

### 阶段四：实现 Probe

实现：

- GAP Linear；
- GAP MLP；
- 2-layer Transformer；
- CLS head；
- Query-conditioned head；
- Functional-program query encoder。

所有 tokenizer 使用相同 optimizer、学习率、scheduler、batch size、训练步数和 seed。

### 阶段五：最小任务集

| 任务 | 数据集 | 主要能力 |
|---|---|---|
| 全局分类 | ImageNet 或 COCO | Global readability |
| 条件查询 | GQA | Query addressability |
| 组合推理 | CLEVR | Compositionality |

每项运行：

- GAP Linear；
- GAP MLP；
- Transformer；
- 500、2,000、8,000 steps；
- 3 seeds。

### 阶段六：第一版指标

\[
\Phi_i^{\mathrm{MVP}}
=
[
A_i^{\mathrm{global}},
A_i^{\mathrm{query}},
A_i^{\mathrm{composition}},
G_i^{\mathrm{TF}},
R_i^{\mathrm{comp}},
V_i
]
\]

### 阶段七：Token-budget 实验

统一输出 16、32、64、128 tokens，计算 token-budget curve、AUC 和低预算保持率。

### 阶段八：扩展 Local 与 OCR

仅在前三个任务已经显示预测能力后加入：

- Visual Genome grounding；
- Attribute query；
- Phrase-to-region matching；
- OCR character recognition；
- Small-object analysis。

### 阶段九：预测下游排名

分别计算：

\[
\rho(\Phi^{\mathrm{global}},Y^{\mathrm{caption}})
\]

\[
\rho(\Phi^{\mathrm{query}},Y^{\mathrm{VQA}})
\]

\[
\rho(\Phi^{\mathrm{composition}},Y^{\mathrm{counting}})
\]

\[
\rho(\Phi^{\mathrm{overall}},Y^{\mathrm{overall}})
\]

---

## 11. 最小可行版本

### Tokenizer

使用当前已有的全部 tokenizer，例如：

- VILA-U；
- UniTok；
- TokLIP-S；
- TokLIP-L；
- MetaCLIP；
- UniFlow；
- 其他新增连续或离散 tokenizer。

### 数据规模

| 数据集 | 建议规模 |
|---|---:|
| ImageNet | 100k–200k 图像 |
| GQA Balanced | 30k 问题 |
| CLEVR | 20k 问题 |

### 第一阶段输出表

| Tokenizer | Global AULC | Query AULC | Composition AULC | TF Gain | Comp. Retention | Stability |
|---|---:|---:|---:|---:|---:|---:|
| VILA-U |  |  |  |  |  |  |
| UniTok |  |  |  |  |  |  |
| TokLIP-S |  |  |  |  |  |  |
| TokLIP-L |  |  |  |  |  |  |
| MetaCLIP |  |  |  |  |  |  |

---

## 12. 必要消融实验

### Transformer 容量

比较 1、2、4 layers，判断排名是否依赖 probe 容量。

### Projector 类型

比较 Linear、MLP、Low-rank MLP。

### Query 表示

比较结构化 query、functional-program tokens 和无 query。

### Token 顺序

比较原始顺序、随机打乱和移除位置编码：

\[
G_i^{\mathrm{order}}
=S_i^{\mathrm{original}}-S_i^{\mathrm{shuffled-order}}
\]

### 输入分辨率

比较 native、common 和实际 MLLM 接入分辨率。

### 训练预算

比较 Final Accuracy 与 AULC 在 500、2,000、8,000 steps 下的排名稳定性。

---

## 13. 成本统计

记录：

- 特征提取时间；
- 特征存储空间；
- Probe 训练时间；
- GPU 显存；
- Trainable parameters；
- FLOPs；
- 达到指定性能阈值所需的训练步数。

---

## 14. 项目成立标准

项目至少需要满足：

1. Transformer compatibility 指标与下游 MLLM 排名稳定相关；
2. 预测能力优于 linear probing、TokBench、reconstruction 和 retrieval；
3. 控制全局语义能力后，仍能解释 counting、relation、local grounding、OCR 或 token compression 的差异；
4. 结论在 leave-one-tokenizer-family-out 下成立；
5. 评测成本显著低于完整 MLLM 微调。

最终希望验证：

\[
\text{MLLM Performance}
\not\approx
\text{Semantic Quality Only}
\]

而更接近：

\[
\text{MLLM Performance}
\approx
f(
\text{Semantic Readability},
\text{Query Addressability},
\text{Compositionality},
\text{Local Addressability},
\text{Token Efficiency}
)
\]

本项目的核心产出是一个**不依赖语言模型、统一使用小型 Transformer、能够低成本预测 visual tokenizer 下游 MLLM 排名的兼容性 benchmark**。
