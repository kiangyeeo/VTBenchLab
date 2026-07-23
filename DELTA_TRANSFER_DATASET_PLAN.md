# DeltaTransfer（暂名）：视觉 tokenizer 干预几何与下游迁移预测

> 调研与落地方案，2026-07-23  
> 重点：新颖性边界、CLEVR 替代路线、数据生成协议、指标修正、实验与停止条件  
> 说明：`FactorBench` 已被同名[量化投资产品](https://factorbench.com/)使用，[`InterveneBench`](https://arxiv.org/abs/2603.15542) 也已被 2026 年的因果推理 benchmark 使用，因此本文只用 `DeltaTransfer` 作为工作名，投稿前还需正式查重。

---

## 0. 结论先行

### 0.1 可以换掉 CLEVR，但不应换成另一个单一数据集

最稳妥的方案不是寻找一个 “CLEVR-but-better”，而是建立分层数据栈：

1. **公开数据 smoke test**：MPI3D / 3D Shapes 验证配对、切分、统计与代码正确性。
2. **低成本精确核心**：基于 Spriteworld 自建 `FactorWorld-2D`，快速生成严格反事实图。
3. **论文主数据**：基于 Kubric 自建 `FactorWorld-3D`，分为 primitives exact tier 与 GSO realistic tier。
4. **OCR 独立域**：改造 SynthTIGER，使其接受显式 `scene_spec` 并输出严格交叉反事实。
5. **外部泛化**：Causal Triplet、VisMin，以及经严格筛选的自然编辑或多视角数据；外部分数单独报告，不与精确合成数据混成一个总分。

这样做的价值不只是“视觉更真实”，而是能够同时检验：

- 同一操作跨场景是否复用；
- 同一操作跨 nuisance 是否稳定；
- 同一指标跨 renderer 是否仍然成立；
- 合成数据上的 tokenizer 排名能否迁移到自然图像和完整 MLLM。

### 0.2 原始 idea 的数学核心已有直接撞车，必须改主张

[Learning Robust Intervention Representations with Delta Embeddings（ICLR 2026）](https://iclr.cc/virtual/2026/poster/10011440) 已直接使用

\[
\delta_a(x)=\phi(x_{\text{after}})-\phi(x_{\text{before}})
\]

并要求同一 action 的 delta 跨 object/scene 稳定、不同 action 可分、与无关场景因素解耦。原方案的 “同 factor 编辑方向一致 + 不同 factor 分离” 因此不能作为主要新颖性。

原始 TCS 的平均 delta cosine 也与 [Geometry of Abstraction](https://pmc.ncbi.nlm.nih.gov/articles/PMC8451959/) 中的 Parallelism Score 同属一个指标家族；该工作还说明，方向平行不自动等价于跨条件可读性，因此必须同时测 CCGP。

这个项目还值得做，但主问题应改成：

> 在不训练完整 MLLM 的前提下，冻结视觉 tokenizer 的、分层且位置对齐的干预几何，能否在未见 tokenizer family 上，增量预测受控 MLLM 的任务表现或视觉 token 压缩退化？

成立的贡献不是“首次发现语义编辑对应方向”，而是以下连接：

- **training-free / low-cost tokenizer 诊断**；
- **逐层、逐粒度、逐 factor 的能力谱**；
- **从 translation direction 升级到可复用的低复杂度 operator**；
- **跨 renderer、跨自然域的泛化**；
- **控制 AC、GW、RankMe、Rank-e、linear probe 等基线之后，对 MLLM 排名仍有增量预测力**。

### 0.3 最重要的先后顺序

不要先花数月生成几十万张图。先用公开数据和 12–20 个 encoder 做一个 go/no-go：

1. 实现 directed operation、whitened parallelism、CCGP、low-rank operator 与 interaction residual；
2. 验证这些量在 held-out context / renderer 上不是随机高维、mask 面积或渲染痕迹造成的；
3. 验证它们能否在 leave-one-family-out 下，超过简单 pair distance、linear probe、RankMe 等基线；
4. 只有出现稳定增量信号，才投入 Kubric 与 OCR 主数据生成。

---

## 1. 新颖性与文献边界

### 1.1 与原方案最接近的工作

| 工作 | 已经做了什么 | 对本项目的约束 |
|---|---|---|
| [Causal Delta Embeddings, ICLR 2026](https://iclr.cc/virtual/2026/poster/10011440) | 用干预前后差向量表示 action；约束同 action 跨场景一致；在 Causal Triplet 上做 action OOD 分类 | 不能声称首次提出 delta intervention、跨场景一致或 factor separation；应作为最强概念基线 |
| [Geometry of Abstraction](https://pmc.ncbi.nlm.nih.gov/articles/PMC8451959/) | Parallelism Score 测同一变量跨 context 的 coding-direction 平行性；CCGP 测跨 context 解码 | TCS 应明确归入 PS 家族，并与 CCGP、noise-aware metric 配套 |
| [Transformation Learning, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/972cd27c994a806e187ef1c2f5254059-Abstract-Conference.html) | 从 image pair 学与图像内容无关的 transformation representation | 不能把 “同 transformation 跨图稳定” 单独作为贡献；差异应落在 frozen tokenizer model selection |
| [Causal Triplet](https://proceedings.mlr.press/v213/liu23a.html) | 以干预前后图像预测 action，并提供 ProcTHOR / EPIC-KITCHENS 的组合与系统 OOD | 是可直接复用的外部验证集，也是必须比较的数据先例 |
| [Formalizing the Binding Problem](https://arxiv.org/abs/2606.03976) | 用 ColorShape、CLEVR、自然数据和冻结 ViT probe 研究 object–feature binding | 颜色/形状/binding 不能再包装成首次；需要证明对 tokenizer 压缩或 MLLM 排名有新增价值 |
| [MMVP / Eyes Wide Shut, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Tong_Eyes_Wide_Shut_Exploring_the_Visual_Shortcomings_of_Multimodal_LLMs_CVPR_2024_paper.html) | 发现 CLIP-blind pairs，并连接视觉 encoder 盲点与 MLLM 失败 | “encoder 盲点传到 MLLM”已有先例；本项目必须比 pair sensitivity 更具体、更可预测 |
| [Rank-e, EACL 2026](https://aclanthology.org/2026.eacl-long.146/) | 用视觉/文本 encoder 的实体先验排序预测 MLLM encyclopedic VQA | 是“廉价 encoder proxy → MLLM”最直接基线之一 |
| [AC Score / Law of Vision Representation](https://arxiv.org/abs/2408.16357) | 用 alignment 与 correspondence 预测 MLLM 视觉表征质量 | 必须报告在 AC 之后的增量价值 |
| [GW model selection, CVPR 2026](https://arxiv.org/abs/2605.01325) | 用 inference-only 的 Gromov–Wasserstein 结构指标预测 VLM encoder 排名 | 不能声称首次低成本选择视觉 encoder；GW 是强基线 |
| [RankMe](https://proceedings.mlr.press/v202/garrido23a.html)、[LogME](https://proceedings.mlr.press/v139/you21b.html) | 无标签有效秩、feature-label evidence 预测迁移质量 | 必须证明复杂干预几何不是 generic feature quality 的替代写法 |

其他需要覆盖的边界：

- [Lenc & Vedaldi](https://arxiv.org/abs/1411.5908) 已用 transformation layer 衡量视觉 representation 的 equivariance/equivalence。
- [3DIEBench](https://proceedings.mlr.press/v202/garrido23b.html) 已用可控 3D 变换研究 invariant/equivariant representation。
- [EqBen](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Equivariant_Similarity_for_Vision-Language_Foundation_Models_ICCV_2023_paper.html) 已通过 paired image–caption 的最小变化评估视觉语言 equivariance。
- [Locatello et al.](https://proceedings.mlr.press/v97/locatello19a.html) 说明无监督 disentanglement 不可在缺乏归纳偏置时一般识别，且 disentanglement metric 不自动带来下游 sample efficiency。本项目有已知 intervention supervision，可以绕开该不可能性，但仍必须实证 transfer。
- [DCI-ES](https://iclr.cc/virtual/2023/poster/11402) 与 [MDL probing](https://aclanthology.org/2020.emnlp-main.14/) 提醒我们：不仅要测“能不能解码”，也要测用多少数据和多复杂的 readout 才能解码。

### 1.2 可以保留的安全定位

建议定位句：

> We introduce an intervention-conditioned, layer- and granularity-aware representation-geometry protocol for predicting frozen visual-tokenizer transfer to controlled MLLMs, and test whether it adds value beyond generic model-selection proxies on unseen tokenizer families.

明确不主张：

- 首次用 counterfactual image pairs；
- 首次用 representation delta；
- 首次测同编辑方向一致；
- 首次测 factor disentanglement；
- 首次发现视觉 encoder 质量影响 MLLM；
- 首次低成本预测 VLM/MLLM 排名；
- 恢复真实 SCM 或证明 causal identifiability。

### 1.3 建议预注册的可证伪假设

1. **H1：增量预测**  
   object-aligned、whitened PS + CCGP + operator score，在控制 reconstruction、pair distance、RankMe、Rank-e、AC、GW、linear probe、参数量、输入分辨率后，仍提高 held-out-family 的 MLLM task rank。

2. **H2：factor-task 特异性**  
   OCR 几何应更强地预测文字任务；spatial/relation 应更强地预测空间与组合任务；若所有 factor 对所有任务贡献相同，指标很可能只是在测 generic quality。

3. **H3：geometry 超过 sensitivity**  
   CCGP、operator sample efficiency 应比单纯 \(\|\delta\|\) 或 semantic/nuisance 距离比更能预测下游。若没有，复杂几何没有必要。

4. **H4：跨域成立**  
   合成 renderer 上得到的 tokenizer 排名应在 held-out renderer 和自然 counterfactual 上保持；若明显翻转，只能做 renderer-specific diagnostic。

5. **H5：组合性有独立价值**  
   factor×factor interaction residual 或由单因素拟合 operator 预测双因素编辑的误差，应与未见组合任务表现相关；否则删除 composition claim。

6. **H6：稳定性**  
   结论在层、合法 readout、whitening seed 与合理 gauge 下稳定；若排名高度依赖任意 pooling 或 normalization，不能宣称 tokenizer 本体能力。

---

## 2. 与仓库现有主线的关系

仓库已经有三个高度相关的设计：

- [BindCurve](./BINDCURVE_MAINLINE.md)：绑定信息、token budget \(K\)、监督预算 \(b\) 与压缩退化；
- [CE-Break](./CE_BREAK_MAINLINE.md)：semantic/nuisance 干预与 whitening；
- [CVLC](./CVLC_README.md)：受控 pair、数据 schema、切分、审计和低成本迁移预测。

新 idea 不宜发展为与它们平行且重复的第四套 benchmark。更清晰的架构是：

```text
共享 intervention-graph 数据引擎
├── CVLC：少量监督下的任务可迁移性
├── CE-Break：semantic vs nuisance 的可分性
├── BindCurve：binding 在 K / b / difficulty 下的崩塌曲线
└── DeltaTransfer：同一干预规律跨 context / renderer 的低复杂度复用
```

DeltaTransfer 真正新增的部分应限定为：

1. directed operation，而不是粗粒度 factor 平均；
2. semantic×nuisance crossed graph；
3. held-out-context / held-out-renderer operator reuse；
4. factor×factor interaction；
5. 这些指标对受控 MLLM 排名的增量预测。

### 2.1 下游预测目标的优先级

按可识别性与成功概率排序：

1. **主目标**：同一 encoder、LLM、connector、训练集，只改变 compressor 或 token budget \(K\)，预测下游退化 \(\Delta Y\)。
2. **次目标**：固定 LLM、connector、数据和训练配方，预测不同视觉 tokenizer family 的任务条件排名。
3. **外部验证**：解释公开 MLLM 的差异。
4. **暂不主张**：用一个 intrinsic score 预测任意公开 MLLM 的综合榜单。

公开 MLLM 同时改变视觉塔、connector、LLM、训练数据和分辨率，最多只能作为 ecological validity，不能作为主要因果证据。

---

## 3. CLEVR 替代数据源比较

选择标准不只是“看起来更真实”，而是：

| 维度 | 必须回答的问题 |
|---|---|
| 干预精度 | 能否从同一个 latent scene 重新渲染，只改变白名单字段？ |
| 多对象与绑定 | 能否固定 attribute marginals，只交换 object–attribute assignment？ |
| nuisance 正交性 | 能否独立控制 camera、lighting、background、texture、compression？ |
| 标注 | 是否有 instance mask、object ID、pose、depth、relation graph、字符框？ |
| 可复现 | 是否能保存完整 scene spec、renderer 版本与资产 hash，而不只保存 seed？ |
| 跨域 | 是否能提供第二 renderer 或自然外部验证，排除 renderer shortcut？ |
| 许可 | 代码、模型、资产、字体和背景是否都能逐项发布 license manifest？ |
| 成本 | 是否可以先小规模验证，再扩到每个 directed transition 500–1000 个独立 context？ |

### 3.1 候选结论

| 候选 | 优点 | 主要缺点 | 定位 |
|---|---|---|---|
| [Spriteworld](https://github.com/google-deepmind/spriteworld) | 多物体、颜色、形状、尺寸、位置、角度可程序控制；Apache-2.0；生成极快 | 视觉简单，没有真实相机、灯光和材质 | **最佳 2D 精确核心** |
| [ShapeWorld](https://github.com/AlexKuhnle/ShapeWorld) | 内置 world model、颜色/形状/空间关系/数量/语言逻辑；MIT | 依赖较旧，需改造成同一 world 的显式干预重渲染 | 关系和 caption 逻辑参考 |
| [MPI3D](https://github.com/rr-learning/disentanglement_dataset) | toy/realistic/real 三域，完整因素组合，能直接索引严格单因素 pair | 单物体、64×64，无 count/binding；公开数据可能进入预训练语料 | **最快 smoke test** |
| [3D Shapes](https://github.com/google-deepmind/3d-shapes) | 480K 全组合，因素索引清楚 | 单物体、64×64、因素较窄 | 单元测试 |
| [Causal3DIdent](https://github.com/ysharma1126/ssl_identifiability) | 连续 nuisance 与因果依赖，分辨率和外观优于传统 disentanglement 数据 | 单物体；公开样本并非为每个样本提供严格一对一 OAT counterpart | 公共压力测试 |
| [3DIEBench](https://proceedings.mlr.press/v202/garrido23b.html) | 大规模受控 3D transformation，直接面向 equivariance | 主要是 pose、illumination、deformation，不覆盖完整 binding/OCR | go/no-go 公共基线 |
| [Kubric](https://github.com/google-research/kubric) | Blender + PyBullet；可控制资产、材质、数量、3D 布局、相机、灯光、HDRI；可输出 mask/depth/flow/metadata | 渲染与工程成本高；资产许可需单独审核 | **最佳 3D 论文主核心** |
| [MOVi](https://github.com/google-research/kubric/blob/main/challenges/movi/README.md) | MOVi-C 含真实 GSO 资产与 HDRI，可参考成熟配置 | 固定下载集不是为本项目的反事实 graph 设计 | 复用配置，不直接当主数据 |
| [ProcTHOR](https://github.com/allenai/procthor) / [AI2-THOR](https://ai2thor.allenai.org/ithor/documentation/object-state-changes/) | 室内对象状态丰富：open/close、toggle、dirty/clean、fill、break 等 | shape swap 与严格正交控制弱于 Kubric | state/action 外部域 |
| [Causal Triplet](https://github.com/CausalTriplet/causaltriplet) | 已提供 ProcTHOR 与 EPIC-KITCHENS 干预对、组合 OOD | 与 Causal Delta Embeddings 高度相关，数据体量大 | 最强 action 外部验证 |
| [CLEVRTex](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/e2c420d928d4bf8ce0ff2ec19b371514-Abstract-round2.html) | 纹理、材质与背景更复杂 | 仍是 CLEVR 系；不能解决新颖性问题 | 仅纹理压力测试 |
| [SynthTIGER](https://github.com/clovaai/synthtiger) | 文本、字体、字距、背景、透视、模糊等可控；MIT | 必须改为显式 scene spec；字体/背景许可需逐项审核 | **OCR 主核心** |
| [TRDG](https://github.com/Belval/TextRecognitionDataGenerator) | 快、易改、MIT | metadata 和 layout 控制弱于 SynthTIGER | OCR 第二 renderer |
| [VisMin](https://arxiv.org/abs/2407.16772) | 自然图像 object/attribute/count/spatial minimal changes，含人工质量控制 | 编辑不能保证像素级单变量，不能充当精确因果核心 | **自然外部验证** |
| [MagicBrush](https://osu-nlp-group.github.io/MagicBrush/) | 自然图像局部编辑、source/target/instruction/mask | 生成式编辑会引入 collateral changes 和生成器痕迹 | 只报告外部 rank/sign agreement |
| [uCO3D](https://github.com/facebookresearch/uco3d) | 同一真实物体多视角，含 camera/mask/3D 信息 | 视角与背景、曝光、轨迹共同变化 | 真实 camera nuisance 辅助集 |

### 3.2 推荐的数据栈

#### Tier 0：公开 sanity

- MPI3D：验证索引、directed edge、反向平衡、whitening、bootstrap；
- 3DIEBench / Causal3DIdent：验证连续变换、层选择与 subspace；
- Causal Triplet：验证 action/state 和组合 OOD。

这些数据便于快速发现指标无效，但不应承担最终“无污染、跨 renderer”的结论。

#### Tier 1：FactorWorld-2D

用 Spriteworld 的 renderer，自建稳定的 scene graph、instance mask、operation 和 split。其作用是：

- 穷举 factor value grid；
- 大规模 negative control；
- 测试 binding-swap、relation control 和统计功效；
- 作为 Kubric generator 的逻辑 oracle。

视觉简单不是致命问题，因为它不承担 realism claim；它承担的是 exactness 与 coverage。

#### Tier 2：FactorWorld-3D

用 Kubric 构建两个域：

1. **KuBasic exact**：规则几何体、可控纯材质、固定渲染参数，承担颜色、形状、材质、数量、位置、关系和 binding 的主测量。
2. **GSO realistic**：真实扫描物体与 HDRI，承担位置、数量、关系、相机、光照和 object-state 风格的迁移验证。

GSO 纹理物体不适合被强行涂成纯蓝后再宣称“只改了自然物体颜色”。这类样本最多是 stress test。GSO 的资产许可与 Poly Haven 等背景资产必须逐项写入 manifest；不能因 Kubric 代码为 Apache-2.0，就假设全部资产也自动是 Apache-2.0。

#### Tier 3：FactorText

改造 SynthTIGER，使每个样本由显式 `scene_spec` 驱动：

- semantic：字符替换、数字替换、相邻字符交换、单词替换；
- nuisance：字体、字号、weight、kerning、背景、透视、旋转、模糊、JPEG；
- 标签：transcript、字符框、glyph mask、line/word ID；
- 许可：只使用 SIL OFL 或同等明确许可字体，并保存字体文件 hash。

TRDG 可作为第二 OCR renderer，验证方向是否来自 SynthTIGER 的模板痕迹。

#### Tier 4：自然外部验证

优先次序：

1. VisMin：object、attribute、count、spatial；
2. Causal Triplet / EPIC-KITCHENS：state/action；
3. MagicBrush 第一轮、局部且 mask 外变化小的子集；
4. uCO3D：同一真实对象的 camera-view nuisance。

这些数据不可与 exact tiers 混成一个总分。应分别报告：

- exact-domain score；
- held-out-renderer score；
- natural-domain rank agreement；
- natural-domain calibration failure。

---

## 4. 核心数据结构：intervention graph，而不是独立 pair

### 4.1 Latent scene

每个基础 context 保存完整 blueprint：

\[
c=(\text{renderer},\text{asset IDs},\text{objects},\text{attributes},
\text{layout},\text{camera},\text{lights},\text{background},\text{render settings})
\]

一个 semantic operation 必须是有方向且可执行的：

\[
g=(\text{factor},\text{source value},\text{target value},
\text{target object/role},\text{magnitude})
\]

例如：

```text
color:red>blue@object_17
shape:circle>square@object_04
position:cell_2>cell_5@object_09
count:3>4@class=small_sphere
binding:swap_color@object_02,object_08
text:A>E@char_slot_3
state:closed>open@cabinet_12
```

不能把所有颜色操作先混成一个 `color direction`。`red→blue` 与 `blue→red` 应先作为相反的 directed operation 分开计算，`red→blue` 与 `green→yellow` 也不应被假定为同一向量。

### 4.2 Semantic×nuisance crossed design

最低数据单元是一个 \(2\times2\)：

\[
\begin{aligned}
x_{00}&=R(s,n), \\
x_{10}&=R(do_f(s),n),\\
x_{01}&=R(s,do_q(n)),\\
x_{11}&=R(do_f(s),do_q(n)).
\end{aligned}
\]

它允许计算同一 semantic edit 在两个 nuisance condition 下的变化：

\[
\delta_f(n_0)=h(x_{10})-h(x_{00}),\qquad
\delta_f(n_1)=h(x_{11})-h(x_{01}).
\]

以及 difference-in-differences：

\[
I_{f,q}=h(x_{11})-h(x_{01})-h(x_{10})+h(x_{00}).
\]

如果只生成 \(x_{00},x_{10}\)，无法区分“真正跨 context 复用的语义规律”和“某个固定背景/渲染边缘的 shortcut”。

### 4.3 用 factor grid 复用节点

不必为每条 edge 独立渲染一对图。对于一个 factor：

- 取 \(L\ge4\) 个 semantic values；
- 取 \(J\ge3\) 个 nuisance states；
- 每个 context graph 渲染 \(L\times J\) 个 node；
- 从 node graph 派生同 nuisance 下的所有 directed semantic edges；
- 从平行边派生跨 nuisance consistency。

例如 \(L=4,J=3\) 时，一个 graph 只需 12 张图，却能得到每个 ordered transition 在三个 nuisance 下的平行边。需要注意：

> 边共享 node，因此统计学样本量仍然是独立 graph 数，不是 \(L(L-1)J\) 条边数。

多因素不做完整笛卡尔积。主数据每个 graph 只展开一个 factor grid；另抽 10%–20% graph 渲染 factor×factor 的 \(2\times2\) square，专门测交互。

### 4.4 第一版因素与干预

| 因素 | 主干预 | 必须固定或匹配 | 阳性/阴性控制 |
|---|---|---|---|
| color | target object 的有向 albedo 替换 | object ID、shape、pose、mask area、camera、light | approximately isoluminant palette；全局亮度变化作为 nuisance |
| shape | 同 object role 的 shape 替换 | centroid、3D/2D size、投影面积、orientation | area-matched vs area-mismatched 两轨 |
| material | diffuse↔metallic / roughness level | geometry、albedo family、camera、light | 只改 light 的 nuisance |
| position | target 在预定义安全 cell 间移动 | object identity、位移长度、可见度、遮挡层级 | 相同位移但不翻转关系的 matched-motion |
| count | \(n\leftrightarrow n+1\) | 单体尺寸、密度、总前景面积 strata、遮挡难度 | same-count replacement；增减方向平衡 |
| relation | 两物体对称换位使 left/right 或 above/below 翻转 | 中点、间距、总位移、属性集合 | 相同总 motion 但 predicate 不翻转 |
| binding | 两物体交换颜色/形状/材质 | 属性 bag、数量、布局、每个属性边际完全不变 | bag-of-attributes representation 应失败 |
| OCR content | 宽度匹配的字符/数字替换或字符交换 | slot、字符数、版式、近似 glyph 面积 | font/background/blur 只作 nuisance |
| state/action | open/close、on/off、full/empty 等 | target ID、viewpoint、非目标物体 | Causal Triplet / ProcTHOR 外部验证 |

### 4.5 三个容易被误称为“单变量”的因素

#### Relation

`left_of(A,B)` 是位置的函数，不能在位置完全不变时单独执行 `do(relation)`。正确做法是：

- 以两物体中点为中心做对称交换；
- 保持物体间距、总位移、大小、颜色、数量不变；
- 另生成相同总位移、但关系不翻转的 matched-motion control。

可定义 control-corrected relation effect：

\[
\Delta_{\text{relation}}
=\Delta_{\text{relation-flip motion}}
-\Delta_{\text{matched non-flip motion}}.
\]

#### Count

增加一个物体必然改变前景面积、遮挡和阴影。这些是允许的 causal descendants，而不是 generator bug。需要：

- 把 descendants 显式写入 operation schema；
- 对 mask area、遮挡率和新增位置分层；
- 加 same-count replacement 与面积匹配控制；
- 不宣称 pixel-level “只改变 count”。

#### Binding

单对象 recolor 只证明 attribute sensitivity，不证明 binding。主 binding 操作应交换两个对象的属性：

```text
before: red circle + blue square
after:  blue circle + red square
```

颜色集合、形状集合、数量和布局完全不变。一个完美的 bag-of-attributes 表示可能在普通 recolor 上获得很高 TCS，却在该 swap quartet 上完全失败。因此 binding 必须有单独的 target/distractor locality 与下游任务。

### 4.6 Nuisance 设计

语义因素与 nuisance 不应使用同一个 “方向一致” 目标。一般预期是：

- semantic edit：可见、结构化、跨 context 可复用；
- nuisance edit：尽量低敏感，或至少不破坏 semantic operator；
- renderer：作为更高层 domain shift，不要求逐像素对齐。

第一版 nuisance：

| 类别 | 变量 |
|---|---|
| camera | azimuth、elevation、distance、focal length 的安全区间 |
| lighting | azimuth、intensity、color temperature、HDRI |
| background | 纯色、纹理、HDRI、clutter level |
| appearance | 非目标对象 texture/material |
| visibility | 轻遮挡、目标面积、depth order；需分 difficulty |
| postprocess | blur、noise、JPEG、resize kernel |
| OCR | font、weight、kerning、perspective、background、blur |

所有 nuisance 取值都要先通过 observability audit，避免“语义改变在某些 condition 下肉眼不可见”。

---

## 5. 数据 schema 与生成器接口

### 5.1 不只保存 pair 表

建议使用六张 Parquet 表：

```text
graphs.parquet
nodes.parquet
operations.parquet
edges.parquet
quartets.parquet
assets.parquet
```

图像、mask、depth 等使用对象文件或 WebDataset shard。

#### graphs

```text
graph_id
scene_family_id
renderer_name / renderer_version / renderer_commit
blueprint_uri / blueprint_sha256
split
split_axes
generator_seed
render_config_json
```

#### nodes

```text
node_id / graph_id
semantic_state_json
nuisance_state_json
image_uri / image_sha256
instance_mask_uri / amodal_mask_uri
depth_uri / normal_uri
scene_graph_json
pixel_statistics_json
visibility_statistics_json
```

#### operations

```text
operation_id
operation_key
factor
source_value / target_value / magnitude
target_object_ids / target_roles
allowed_latent_diff_json
allowed_causal_descendants_json
inverse_operation_id
```

#### edges

```text
edge_id / graph_id / operation_id
source_node_id / target_node_id
nuisance_id
target_mask_uri
changed_pixel_ratio
outside_mask_change
difficulty_bin
visible_flag
```

#### quartets

```text
quartet_id / graph_id
factor_operation_id / nuisance_operation_id
node_00 / node_10 / node_01 / node_11
factor_pair_id
```

#### assets

```text
asset_id / asset_sha256
source_url / local_uri
asset_type
license / attribution / redistribution_allowed
font_family / font_license
```

只保存 seed 不够。依赖升级、资产版本变化或随机调用顺序改变后，相同 seed 未必重现相同场景；必须保存完整 blueprint、版本和 hash。

### 5.2 生成器的最小接口

```python
blueprint = sample_blueprint(scene_family_id, rng)
factor_grid = enumerate_factor_states(blueprint, factor_spec)
nuisance_grid = enumerate_nuisance_states(blueprint, nuisance_spec)

for factor_state in factor_grid:
    for nuisance_state in nuisance_grid:
        scene = instantiate(blueprint, factor_state, nuisance_state)
        latent_diff_audit(scene, blueprint, whitelist)
        outputs = render(scene, render_config)
        semantic_oracle(outputs, scene)
        write_node(outputs, scene)

derive_edges_and_quartets(graph_id)
run_artifact_audits(graph_id)
```

关键原则：

- source 与 target 都从 latent graph 重新渲染，不使用 inpainting 作为 exact tier；
- 同一 graph 的所有 node 使用锁定的 renderer 配置；
- semantic direction、target object、位置、颜色边际、source/target 出现频率必须平衡；
- 文件名和目录不能泄露 factor 或 source/target 标签给下游模型。

---

## 6. 表征抽取与指标修正

### 6.1 什么情况下 delta 才合法

只允许在同一个 tokenizer setting 内相减：

\[
m=(\text{checkpoint},\text{layer},\text{surface},\text{preprocess},
\text{resolution},\text{token policy},\text{compressor},K).
\]

需要满足：

- 相同 feature basis；
- 相同层与 normalization 位置；
- 相同 factor-specific canonicalizer；
- 相同 token 数或已被合法映射为固定结构。

禁止：

- 跨 tokenizer 直接相减向量；
- 跨 layer、pre-LN/post-LN 或不同 compressor 直接相减；
- 对 VQ code ID 做数值减法；
- 将长度、顺序不同的 raw token matrix 直接相减。

离散 tokenizer 应使用 codebook lookup 后的 embedding 或官方 post-quant representation。跨模型只比较各自空间内得到的无量纲标量与 OOF 预测性能。

### 6.2 因素特定的 token canonicalization

raw flattened patch delta 会把“编辑发生在不同 slot”误判为方向不一致；只用 CLS/mean 又会抹掉位置、OCR 和 binding。建议三粒度分别报告：

1. global CLS / mean；
2. ROI-aligned local representation；
3. structured token set / fixed slots。

推荐规则：

| factor | canonical representation |
|---|---|
| color / material / shape | before/after union mask 覆盖的 ROI；固定 2×2 或 4×4 bins |
| position | 有序 `[source ROI, destination ROI, global]`，不能只做 object-centric crop |
| count | changed-instance union + global summary |
| relation / binding | 有序 `[subject ROI, object ROI, relative coordinates, global]` |
| OCR | fixed character slots 或 glyph-aligned bins |

优先使用数据提供的几何对应；feature-space OT 只作次级对齐。没有 token coordinate 的模型不能参加 local main track，但可以参加 global track。

还必须报告：

- target-region signal；
- distractor-region leakage；
- background/off-target leakage；
- spatial locality 与全局可读性的差异。

### 6.3 不要把所有东西压成一个 TCS

第一版 scorecard 分成五个轴：

1. **Sensitivity / observability**：语义改变是否超过 null rerender 与 nuisance；
2. **Parallelism**：同一 directed transition 跨 context 是否平行；
3. **Abstraction / reuse**：少量 context 学到的规则能否泛化到新 context；
4. **Selectivity / separation**：能否区分 factor 或 directed operation，而不是只靠 magnitude；
5. **Composition**：单因素规律能否预测双因素 representation，交互是否小。

在验证其与下游关系之前，不应手工加权成一个总分。

### 6.4 Observability 与 semantic-over-nuisance

先用 null rerender 和 nuisance edge 建立可见阈值 \(q_{0.95}\)。方向指标只在 visible edge 上计算，同时报告 coverage：

\[
\text{coverage}_g=P(q_{\text{semantic}}>q_{0.95,\text{null}}).
\]

SNR 不建议用不稳定的均值比。可以报告：

\[
\text{PSup}_g=P(q_{\text{semantic}}>q_{\text{nuisance}})-0.5
\]

或配对 log-energy difference。CE-Break 的 COS / eigenspectrum 应作为基线。

### 6.5 Parallelism：TCS 只作为一个受限候选

对同一 directed operation \(g\) 的 unit delta \(u_i\)，无偏平均 pairwise cosine 可写为：

\[
\operatorname{PS}_g=
\frac{\left\|\sum_{i=1}^n u_i\right\|^2-n}
{n(n-1)}.
\]

不能把 \(O(n^2)\) 个 cosine 当成独立样本；推断单位仍是 graph/context。

raw cosine 对可逆的非正交重参数化 \(z'=Az\) 不稳定。主报告至少包括：

- official/raw surface；
- train-only robust channel standardization；
- shrinkage covariance 下的 whitened/Mahalanobis cosine；
- 固定 JL projection（如 256/512 维，多 seed）敏感性。

如果合理 gauge 下 tokenizer 排名 Kendall \(<0.7\) 或频繁翻转，只能做 surface-specific claim。

### 6.6 CCGP / held-out-context decoding

对 directed operation 或 factor 训练固定容量 probe：

- train：部分 context、asset、layout；
- test：未见 context；
- stronger test：未见 ordered transition 或未见 renderer；
- 所有 preprocessing、whitening、threshold 只用 train graph。

报告：

- learning curve；
- sample efficiency / MDL；
- grouped out-of-fold accuracy；
- crossnobis distance 或 permutation-normalized separation；
- magnitude-only、mask-area-only、edit-position-only baseline。

这比“mean factor direction 的 Gram 越接近对角越好”更可靠。不同 factor 不必几何正交；相关因素共享 subspace 并不意味着不可用，高维随机向量也天然近似正交。

### 6.7 从 translation 升级到低复杂度 operator

原 TCS 假设：

\[
h(gx)\approx h(x)+v_g.
\]

但有用 representation 也可能满足：

\[
h(gx)\approx \rho_g(h(x)),
\]

其中 \(\rho_g\) 是 context-independent 的低复杂度变换而不是平移。

对每个 operation 比较嵌套模型：

1. additive translation；
2. diagonal affine；
3. rank-\(r\) affine residual；
4. 小型 MLP 上界。

所有 operator 只在 train contexts 拟合，主指标是在未见 context / asset / renderer 上的 normalized \(R^2\)、prediction error 与达到给定误差所需的样本数。核心问题变为：

> 一个新 context 的编辑后 representation，能否由少量样本学到的共享低复杂度规律预测？

这比“delta 是否永远同向”更符合“简单 connector 能否利用”的动机。

### 6.8 Nuisance 与 factor×factor interaction

同一 factor 在两个 nuisance 下的 normalized interaction residual：

\[
\operatorname{IR}_{f,q}=
\frac{\mathbb E\|\delta_f(n_1)-\delta_f(n_0)\|^2}
{\frac12\mathbb E[
\|\delta_f(n_1)\|^2+\|\delta_f(n_0)\|^2]+\epsilon}.
\]

双 factor square：

\[
I_{f_1,f_2}=
h(x_{11})-h(x_{10})-h(x_{01})+h(x_{00}).
\]

还可以分别拟合 \(\hat\rho_{f_1},\hat\rho_{f_2}\)，在 held-out 双编辑样本上测试：

\[
\hat\rho_{f_2}(\hat\rho_{f_1}(h(x)))
\quad\text{vs}\quad h(f_2(f_1(x))).
\]

重要反例：

- 在同一组三个固定 node 上，\(\delta_{a\to b}+\delta_{b\to c}=\delta_{a\to c}\) 是代数恒等式，不是 representation composition 证据；
- 同一固定 pair 上 \(\delta_{a\to b}=-\delta_{b\to a}\) 也是恒等式；
- 只有跨 context 拟合的 operator 预测，或真正的 factor×factor factorial square，才提供非平凡证据。

---

## 7. 切分、审计与防泄漏

### 7.1 最基本的切分单位

所有共享 latent blueprint 的：

- nodes；
- semantic/nuisance edges；
- inverse edges；
- quartets；
- factor×factor squares；

必须位于同一 split。先按 `scene_family_id` 分 graph，再派生 edge；绝不按最终图片或 pair 随机切分。

自然数据：

- MagicBrush 按 source image / edit session；
- uCO3D 按 physical object / video；
- Causal Triplet 按 room / scene；
- OCR 按 lexical base、font family 和 template family。

### 7.2 分层 OOD split

至少保留：

1. **calibration**：拟合 whitening、operator、threshold；
2. **IID diagnostic**：只排查工程错误，不作为主结论；
3. **composition OOD**：未见 shape-color、object-layout、factor combination；
4. **asset OOD**：未见 asset ID / object category；
5. **transition OOD**：未见 ordered source→target；
6. **difficulty OOD**：未见 cardinality、目标尺寸、遮挡级别；
7. **renderer OOD**：Spriteworld→Kubric、KuBasic→GSO、SynthTIGER→TRDG；
8. **natural external**：VisMin / Causal Triplet / filtered natural edits。

intrinsic proxy 数据与完整 MLLM ground-truth 数据最好不共享：

- image；
- scene seed / blueprint；
- asset；
- question template；
- renderer family。

至少一个下游 target 应来自自然图像，否则“预测 MLLM”可能只是同一生成器上的共同过拟合。

### 7.3 自动审计

每个 graph 必须通过：

1. **latent diff**：实际 JSON diff 等于白名单；causal descendants 被显式记录；
2. **determinism**：同 scene spec 重渲染一致，或在已声明数值容差内；
3. **semantic oracle**：颜色/数量/关系/位置/transcript 确实按预期改变；
4. **visibility**：目标面积、出界、遮挡、碰撞符合阈值；
5. **collateral change**：target mask、阴影带与遮挡带之外的 changed-pixel ratio/SSIM/LPIPS；
6. **balance**：source/target、方向、目标位置、object role、答案位置平衡；
7. **duplicate**：image/scene/asset hash 去重；
8. **license**：每个资产、字体、背景都有可追溯许可。

建议验收门槛：

- human/oracle 双侧正确率 \(>95\%\)；
- question-only 接近机会水平；
- mask 外或 edit-side artifact classifier AUC \(\le0.55\)；
- null rerender 的方向分数覆盖 0；
- 每个 operation、difficulty、renderer 人工抽检 1%–2%，且总量至少 500–1000 个独立 graph。

mask 外 side classifier 很重要：所有编辑共享同一种 renderer seam 或局部边缘变化时，TCS 会出现很高的假阳性。

---

## 8. Tokenizer 与 MLLM 实验设计

### 8.1 representation surface 分轨

主结果不能混合以下 surface：

- dense spatial tokens；
- official downstream selected layer；
- compressed tokens；
- pre-quant embedding；
- post-quant/codebook embedding；
- LLM-projected tokens。

建议：

- 以“实际接入 MLLM 的官方视觉层/表面”为主；
- dense、compressed、pre/post-quant、LLM-projected 分 leaderboard；
- native-\(K\)、fixed-\(K\)、equal-FLOPs 分轨；
- layer sweep 是 construct-validity 检查，不是普通附录消融。

一个完美表示在 pre-LN 可能满足平移规律，但 post-LN 的差向量会因基准 \(h(x)\) 而变化；反过来，channel scaling 也能任意改变 cosine 而不改变线性可读性。因此 normalization sensitivity 直接决定 claim 是否成立。

### 8.2 模型数量

当前原始计划的 5 个 tokenizer 只适合 smoke / construct validation，不能支撑跨 family 排名结论。

仓库 [Tokenizer setup](./Tokenizer_set_up.md) 已包含大量 MetaCLIP、SigLIP2 和离散 tokenizer 配置，但“配置数”不等于“独立 family 数”。建议：

- debug：2 个设置；
- go/no-go：12–20 个公开 encoder / tokenizer，覆盖不同 objective 与至少 6 个 family；
- exploratory：至少 16 个设置、6 个 family；
- 论文级 rank prediction：24–30 个设置、至少 8 个 family；
- 外层测试最好每次 hold out 整个 family，最终至少有 4 个从未参与调参的 family。

同 family 的不同 resolution、layer、\(K\) 和 compressor 很有价值，但主要用于“预测受控退化”，不能虚增跨 family 的独立样本量。

### 8.3 完整 MLLM ground truth

必须固定：

- LLM；
- connector architecture 与参数预算；
- instruction/pretraining data；
- optimizer、steps、batch、resolution；
- token budget；
- evaluation prompt 和 decoding。

每个 setting 至少 3 个训练 seed。先检查 ground-truth rank 自身是否稳定；若 seed 间 Kendall \(<0.7\)，任何 intrinsic proxy 都没有稳定目标可预测。

任务按 factor 分组：

| factor score | 主要下游 target |
|---|---|
| OCR | text/document/chart reading |
| count | counting、small-object cardinality |
| spatial/relation | relative position、grounding、relational VQA |
| binding | attribute assignment、multi-object compositional QA |
| state/action | state recognition、action direction |
| generic semantic | object/attribute recognition、caption/VQA |

除 task-specific score 外，再报告综合分，但不允许先看 test 结果再调 factor 权重。

### 8.4 强基线

至少包括：

- reconstruction rFID / PSNR / LPIPS（适用于生成式 tokenizer）；
- raw semantic pair distance 与 semantic/nuisance energy ratio；
- linear probe / few-shot probe；
- RankMe、LogME、TransRate；
- Rank-e；
- AC Score；
- GW structural score；
- CKA / representation similarity；
- CE-Break COS / eigenspectrum；
- CVLC / BindCurve 指标；
- 参数量、feature dimension、token count、输入分辨率、预训练数据规模；
- magnitude-only、mask-area-only、edit-position-only artifact baseline。

真正的贡献判据是：

\[
M_0=\text{architecture/compute confounds},
\]

\[
M_1=M_0+\text{all strong generic proxies},
\]

\[
M_2=M_1+\text{DeltaTransfer factor geometry}.
\]

只有 \(M_2\) 在 held-out family 上稳定改善，才能声称新增预测价值。

### 8.5 统计

- graph / quartet 是抽样单位；
- hierarchical bootstrap：renderer → graph → graph 内 edge；
- tokenizer ranking 使用 nested leave-one-family-out；
- factor 权重、metric choice、layer policy 只在 outer-train families 选择；
- 报告 Spearman、Kendall \(\tau_b\)、pairwise accuracy、top-\(k\) regret、cross-validated \(\Delta R^2/\Delta\tau\)；
- 对 renderer、family 做 cluster bootstrap 或 permutation；
- 不把共享 node 的大量 edge 当作独立样本。

建议 preregister 的论文级目标，不是公认定律，而是内部 stop/go 线：

- metric seed 稳定性 Kendall \(\ge0.8\)；
- 合理 gauge 稳定性 Kendall \(\ge0.7\)；
- held-out-family Kendall \(\tau_b\ge0.45\) 或 Spearman \(\ge0.6\)；
- pairwise ranking accuracy \(\ge70\%\)；
- 相对 \(M_1\) 的 \(\Delta\tau\ge0.10\)，且 family-clustered CI 下界 \(>0\)。

---

## 9. 数据规模与实施路线

### Phase 0：指标去风险，1–2 周

目标：不写大生成器，先判断 idea 是否有信号。

- MPI3D 抽取 10K–20K 严格 indexed pairs / factor grids；
- 加 3DIEBench、Causal3DIdent 或 Causal Triplet 中可直接复用的 3–4 类操作；
- 跑 12–20 个 encoder/tokenizer setting；
- 实现 raw/whitened PS、CCGP、low-rank operator、interaction residual；
- 加 random encoder、shuffle label、magnitude-only、mask-area-only controls；
- 用公开下游结果或一个固定 LLM+connector 的小规模 grid 建初步 target。

**Go**：

- 指标在 held-out context 上显著高于 permutation/null；
- object-aligned 指标不被 global mean 替代；
- whitened/CCGP/operator 至少一项稳定超过 raw distance；
- leave-family-out 出现方向稳定的增量预测。

**No-go / pivot**：

- random feature 同样高；
- 排名随 normalization、层或 seed 任意翻转；
- simple pair distance、RankMe 或 linear probe 完全覆盖信号；
- synthetic→held-out domain 排名系统性反转。

### Phase 1：FactorWorld-2D，2–3 周

第一版先做 color、shape、position、count 四个主 factor，再用一个较小的 binding-swap 子集验证最关键的反例：

- 每 factor \(L=4\) semantic states；
- \(J=3\) nuisance states；
- pilot：每 factor 250 个独立 graph；
- 约 \(4\times250\times4\times3=12{,}000\) 张主图；
- binding-swap 另加约 2,000–3,000 张；
- 10%–20% graph 增加 factor×factor squares；
- 自动生成 mask、relation graph、latent diff 与 audit report。

这里优先验证数据协议、local canonicalization 和 relation/count control，不追求 photorealism。

### Phase 2：FactorWorld-3D / Kubric，4–6 周

先 KuBasic exact，后 GSO realistic。

主 exact tier 建议：

- 4 个主 factor；
- 每 factor 500 个独立 graph；
- 每 graph \(L=4,J=3\)，共约 24,000 张图；
- 6 个 factor pair × 200 graph × 2 nuisance × 4 factorial states，约 9,600 张 interaction 图；
- 合计约 33,600 张 exact render。

扩展：

- 每个主 directed transition 最终 500–1000 个独立 context；
- GSO / HDRI held-out tier 10K–20K；
- 全论文精确合成部分约 50K–100K 图即可，不应在指标未通过 gate 前盲目上到百万规模。

需要锁定：

- Kubric commit；
- Blender 版本；
- Cycles/Eevee、samples、color management；
- camera、light、physics 参数；
- GSO/HDRI/texture 的资产许可与 hash。

### Phase 3：OCR 与外部域，3–4 周

OCR MVP：

- 5K lexical bases；
- 每个 base 2–4 个 content states、3–5 个 nuisance；
- 先做 20K–50K 图；
- 字符替换、数字替换、相邻交换三类 operation；
- held-out font、glyph n-gram、script、background；
- TRDG 作为第二 renderer。

外部验证：

- VisMin 2K–5K 个高质量 pair；
- Causal Triplet 的小规模 action/state 子集；
- MagicBrush 只保留第一轮、局部、mask 外变化小的样本；
- uCO3D 按 physical object 采样 matched view pairs。

### Phase 4：受控 MLLM 排名，4–8 周

先做最可识别的目标：

1. 固定 encoder，只改变 \(K\) / compressor；
2. 固定 LLM+connector，对 12–20 个视觉 setting 跑少量任务；
3. 若 target rank 稳定且 proxy 有增量信号，再扩到 24–30 setting / 8 family；
4. 外层严格 leave-family-out，最后冻结模型后再开 held-out renderer / natural test。

---

## 10. 主要失败模式与预先约定的降级结论

| 结果 | 正确结论或 pivot |
|---|---|
| raw TCS 高但 binding-swap / locality 差 | raw direction 测到 bag-of-attributes 或 renderer artifact；不能称 binding |
| global delta 差、ROI-aligned CCGP 好 | 表示有用但 slot 未对齐；放弃 global-axis claim |
| translation 差、low-rank operator 好 | 改论文主线为 shared operator/subspace，而不是同一方向 |
| 所有 factor 都同样预测所有任务 | 测到 generic quality；保留为通用 proxy，不声称 factor spectrum |
| synthetic 内有效，held-out renderer 失效 | 做 renderer-specific diagnostic；继续改数据前不能声称通用规律 |
| intrinsic 高、MLLM 低 | 视觉信息存在但 connector/LLM 未利用；增加固定预算 projector learning curve，不能把失败全归因 tokenizer |
| 只预测同 family 的 resolution/K | 收缩为 compression-degradation predictor；这仍是可发表且更可信的目标 |
| \(M_2\) 不优于 AC/GW/Rank-e/linear probe | 不发展独立 benchmark；把 intervention geometry 作为 CVLC/BindCurve 的诊断模块 |
| 完整 MLLM seed rank 不稳定 | 先修 ground-truth 实验；不能评价 proxy |

---

## 11. 最小可交付物

### 数据

- versioned scene blueprint；
- graph/node/operation/edge/quartet/asset Parquet；
- exact 2D 与 exact/realistic 3D 两个 renderer；
- OCR renderer；
- hidden renderer 与自然 external split；
- license manifest 与 data card。

### 代码

- generator 与 deterministic replay；
- latent/pixel/artifact audit；
- tokenizer adapter 与 layer/surface registry；
- ROI/slot canonicalizer；
- metric suite；
- hierarchical bootstrap 与 nested LOFO ranking；
- feature cache，不重复抽取 tokenizer 特征。

### 论文证据链

```text
数据 exactness
  → 指标 construct validity
  → held-out context/operator reuse
  → held-out renderer robustness
  → controlled MLLM degradation/rank prediction
  → strong generic baselines 之后的 incremental value
```

这条链中任何一环失败，都应缩小 claim，而不是用更大的数据量掩盖。

---

## 12. 最终建议

1. **现在不要用 CLEVR，也不要直接用 CLEVRTex 作为“换数据集”答案。**
2. **第一周用 MPI3D + 公共 intervention 数据验证指标。**
3. **主数据用 Kubric 自建 latent-scene counterfactual graph；Spriteworld 只承担快速精确核心。**
4. **OCR 用显式 scene-spec 版 SynthTIGER，独立成域。**
5. **自然图像只作 held-out external validation，不与 exact tier 混合。**
6. **把原始 TCS/FSS 改成 directed transition 的 PS + CCGP + low-complexity operator + interaction residual。**
7. **主预测目标先定为同视觉栈的 \(K\)/compressor 退化，再扩到跨 family MLLM 排名。**
8. **5 个 tokenizer 只做 smoke；正式 rank claim 需要 24–30 settings、至少 8 families，并严格 leave-one-family-out。**
9. **只有在 AC/GW/Rank-e/RankMe/linear probe 等基线之后仍有增量价值，才把它发展为独立工作。**

如果严格按这个顺序推进，这个 idea 的风险会从“另一个 CLEVR delta metric”降到一个可证伪、能停止、且与仓库现有主线互补的研究项目。
