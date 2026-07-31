# Tokenizer–Transformer 接口兼容性 Benchmark

## 1. 问题

现有视觉 tokenizer 评测大致分为两类：

- **重建评测**：使用原生 decoder 还原图像，计算 PSNR、LPIPS、rFID、OCR 等指标。
- **下游 MLLM 评测**：将 tokenizer encoder 输出接入 projector 和语言模型，通过完整训练评估 VQA、OCR、caption 等任务。

两者之间存在明显缺口：

1. 下游 MLLM 通常只使用 encoder，不使用 tokenizer 的原生 decoder。
2. 重建结果同时受 encoder 和 decoder 影响，无法单独反映 encoder 质量。
3. 完整 MLLM 训练成本高，不适合作为大规模 tokenizer 筛选方法。
4. linear probing 只能测简单可分性，不能充分反映视觉 token 是否容易被 Transformer 读取和组合。

因此，目标不是再做一个重建 benchmark，而是低成本测量：

> 视觉 tokenizer 的输出能否通过一个轻量接口，被固定 Transformer 有效读取。

---

## 2. 核心设计

对每个待测 tokenizer，仅训练一个受限 projector，其余模块全部冻结。

```mermaid
flowchart LR
    X[输入图像] --> E[冻结的视觉 Tokenizer Encoder]
    E --> Z[异构视觉 Tokens]
    Z --> P[可训练的轻量 Projector]
    P --> U[统一格式的视觉 Tokens]
    U --> T[冻结的 Transformer]
    Q[固定文本 Query] --> T
    T --> H[冻结的输出头]
    H --> Y[答案 / 视觉证据 Tokens]
```

形式化表示为：

\[
\hat y =
H_{\mathrm{fixed}}
\left(
T_{\mathrm{fixed}}
\left(
P_i(E_i(x)), q
\right)
\right)
\]

其中：

- \(E_i\)：第 \(i\) 个视觉 tokenizer 的 encoder；
- \(P_i\)：该 tokenizer 对应的轻量 projector；
- \(T_{\mathrm{fixed}}\)：固定的小型 Transformer；
- \(H_{\mathrm{fixed}}\)：固定语言头或分类头；
- \(q\)：固定格式的任务 query；
- \(\hat y\)：答案 token 或结构化视觉证据。

训练时只更新 \(P_i\)。

---

## 3. 输出形式

最终输出不应是图像，也不应是无监督隐藏向量，而应是带有明确监督目标的结果，例如：

- OCR 字符或文本 token；
- 物体类别与属性；
- 数量；
- 左右、上下、遮挡等空间关系；
- 简短 VQA 答案；
- 区域对应的离散语义标签。

这样可以避免两个问题：

- 输出图像会重新退化为重建 benchmark；
- 输出隐藏向量缺少统一目标，难以直接比较。

---

## 4. Projector 约束

不同 tokenizer 的 token 数量、通道维度和空间结构不同，因此每个 tokenizer 需要独立 projector，但其能力必须严格受限。

建议统一为：

```text
输入维度适配
→ LayerNorm
→ Token-wise Linear
→ 固定插值或轻量 Resampler
→ 统一长度和维度
```

统一输出形状：

\[
P_i(E_i(x))\in\mathbb{R}^{B\times N_0\times d}
\]

需要控制：

- projector 架构；
- 参数量；
- 初始化；
- 训练数据；
- 训练步数；
- 优化器与学习率；
- 输出 token 数 \(N_0\)；
- 输出维度 \(d\)。

若 projector 过强，它可能自行完成大部分视觉建模，使评测失去意义。

---

## 5. 任务划分

### Local Evidence

关注细节信息：

- 场景文字；
- 小物体；
- 局部颜色与属性；
- 字符、数字和符号。

对应下游：

- TextVQA；
- DocVQA；
- ChartQA；
- OCRBench。

### Structural Evidence

关注 token 之间的组合关系：

- 数量；
- 相对位置；
- 区域匹配；
- 表格和文档布局；
- 局部到全局的关系组合。

### Semantic Evidence

关注高层视觉语义：

- 物体和场景类别；
- 简短描述；
- 通用 VQA；
- 属性与关系判断。

---

## 6. 评测指标

不只比较最终准确率，还应比较 projector 的学习效率。

在不同数据预算下训练：

\[
n\in\{1k, 5k, 20k\}
\]

记录：

- Accuracy / ANLS / VQA Score；
- validation loss；
- 达到固定性能所需样本数；
- 性能—数据量曲线下面积。

定义：

\[
\mathrm{InterfaceScore}_i
=
\operatorname{AUC}_{\log n}
\left[
\mathrm{Performance}_i(n)
\right]
\]

该指标衡量：

> 一个 tokenizer 的视觉 token 是否能被固定 Transformer 以较低样本和较小接口成本读懂。

---

## 7. 与完整 MLLM 的验证

Benchmark 的目标不是单独获得高分，而是预测完整 MLLM 下游表现。

实验流程：

```mermaid
flowchart TD
    A[多个视觉 Tokenizer] --> B[Projector-only Proxy Benchmark]
    A --> C[完整 MLLM 训练]
    B --> D[Proxy 排名]
    C --> E[下游任务排名]
    D --> F[Spearman / Kendall 相关性]
    E --> F
```

建议分别验证：

\[
\text{Local Score}
\leftrightarrow
\text{OCR 类下游}
\]

\[
\text{Structural Score}
\leftrightarrow
\text{关系与图表类下游}
\]

\[
\text{Semantic Score}
\leftrightarrow
\text{VQA 与 Caption 类下游}
\]

同时进行：

- 单任务相关性；
- 分组任务相关性；
- 总体排名相关性；
- leave-one-tokenizer-family-out；
- 跨 Transformer 验证。

---

## 8. 创新边界

“冻结视觉 encoder 和语言模型，只训练 projector”本身不是新结构。真正的贡献应放在：

1. 面向异构 visual tokenizer 的统一接口协议；
2. 将 Transformer 可读性定义为独立评测对象；
3. 使用受限 projector 的低样本学习曲线衡量接口兼容性；
4. 系统验证该指标对完整 MLLM 排名的预测能力；
5. 分离局部细节、结构关系和高层语义三类能力。

---

## 9. 一句话定位

> 冻结视觉 tokenizer 与 Transformer，仅训练统一规格的轻量 projector，通过标准化视觉证据任务测量 token 的 Transformer 可读性，并用其低成本预测完整 MLLM 的下游表现。
