# mechanistic/ — 三个机制性角度（不是统计量）

环境：`conda activate TokBench`，加 `TOKENIZERS_PARALLELISM=false`。都在仓库根目录跑。
判决对象统一用 `pe_lang_g14_448` vs `pe_core_g14_448`：同架构、同分辨率、同实验室，
MLLM 差 +6.15 而 ImageNet probing 差 −13.6，是全表最硬的一对。

---

## 1. `rgf.py` — Realizable Gradient Fraction（救梯度的尝试，**已证伪**）

诊断：LESS 的 cosine 比较的是**样本**（固定模型下哪条数据对哪个任务有用）。拿来比较**模型**时，
换 encoder 同时换掉 h 的尺度与几何，跨 encoder 的 cosine 本就不可比。

改法：不看方向，看可实现比例。冻结 LLM，视觉 token 进视觉槽，caption 做 teacher forcing，
反传到视觉槽激活得 G = dL/dh（"LLM 希望视觉 token 往哪动"）。projector 就是线性映射 Z→h，
它的一阶最优更新恰好是把 G 最小二乘拟合到 Z 上，于是
`RGF = 1 - ||G - ZW*||^2 / ||G||^2`（held-out）。是 R²，无量纲，跨 encoder 可比；用随机或已训
projector，不需要训练，不会被"recipe 收没收敛"污染。

    python mechanistic/rgf.py --models pe_lang_g14_448 pe_core_g14_448 \
        --n-images 192 --tokens-per-image 96 --seeds 2

**实测结果：RGF ≈ 0.0000（暖启动 projector 后仍然是 0）**，诊断已排除"常向量主导"
（共享分量占比 0.000）。含义：**LLM 对某个视觉 token 的一阶需求，线性上完全无法由该 token
的内容预测**。这同时解释了 LoRA-B 那版为什么失败——需求里没有内容信息，任何梯度导出的量
（方向或幅度）都分不开 encoder。两种独立功能形式都归零，建议关掉这条线。脚本留作证伪工具。

---

## 2. `sink_intervention.py` — 高范数 sink token 的因果干预

观察：`pe_lang_g14_448` 有 0.998% 的 token 范数 > 该图中位数 5 倍（max/med = 18.2），
这不到 1% 的 token 贡献池化向量 13.6% 的模长，每图 1024 token 的有效秩只有 4.2。
其余 20 个 encoder 基本没有（pe_core max/med 2.30，dinov2_large 1.06）。

假设：sink 对 MLLM 有用（注意力倾倒处），对任何**池化**读出是灾难——而所有便宜 baseline
打分的对象都是池化向量。

    python mechanistic/sink_intervention.py \
        --models pe_lang_g14_448 pe_core_g14_448 dinov2_large siglip2_l16_384 dino_vitb16 \
        --n-images 2000

**实测（600 张图，COCO obj80 mAP）**：

| encoder | sink% | max/med | 全部 token | 去 sink | 只用 sink |
|---|---|---|---|---|---|
| pe_lang_g14_448 | 0.998 | 18.23 | 0.760 | **0.774 (+0.014)** | 0.529 |
| pe_core_g14_448 | 0.977 | 2.30 | 0.724 | 0.723 (−0.001) | 0.518 |
| dinov2_large | 0.781 | 1.06 | 0.727 | 0.726 (−0.001) | 0.480 |

方向对了、唯一有增益的就是有 sink 的那个，但**幅度小**（+0.014 mAP，而 pe_lang 的 ImageNet
probing 缺口是 13.6 分）。所以 sink 是真实机制但不是主因，别当主线卖。

**还没做的那一半（要 2 次训练）**：把 pe_lang 送进 MLLM 的序列里的 sink token 丢掉后重训。
预测 MLLM 下降。若"去 sink → 探针涨、MLLM 降"同时成立，就是一个干净的因果不对称。

---

## 3. `addressability.py` — token 可寻址性（用 LLM 自己的注意力）

到目前为止所有指标量的都是**信息在不在**，没有一个量**LLM 能不能找到它**。MLLM 读图靠注意力检索，
这是 query-key 可分性（几何性质），可以和信息量完全脱钩。

测法：M+1 张图的视觉 token 拼成序列，后接其中一张图的 caption 内容词，跑一次前向拿 attention，
统计 caption 把多少注意力质量放在**自己那张图**的 token 上。随机 = 1/(M+1)。

主指标是**配对差分** `delta = 匹配 caption − 错配 caption`（同图、同干扰项、同槽位，只换 caption），
把一切与内容无关的几何/位置偏置减掉。每条 query 自成一个配对，所以配对差的标准误直接给出误差棒。

    bash mechanistic/run_addressability_full.sh          # 26 个 encoder，约 4 小时
    python mechanistic/analyze_addressability.py mechanistic/out/addressability_full.json

### 已经踩过的三个坑（改配置时别踩回去）

1. **随机投影不行。** 随机投影把视觉 token 放到 LLM 从没见过的区域，delta ≈ 0.0001，纯噪声。
   必须 `--projector-root` 用已训 projector。代价是不再训练无关，但 delta 是同一次 run 内的
   配对对照，收敛差的 projector 给出 delta≈0 而不是系统性偏差，比绝对 loss 稳健。
2. **projector 必须 strict=True 加载。** checkpoint 是 `{"projector": OrderedDict(...), ...}`，
   早期版本读错顶层 key，静默退化成随机 MLP（6 个 key 全 missing）却照常出数。已改成加载失败即报错。
3. **RoPE 近因偏置。** 固定把 own image 放序列最前会系统性压低它的份额（实测 0.0435，随机是 0.1250）。
   脚本已随机化 own image 的槽位。

### 效应量与所需样本（实测，300 query）

| encoder | delta | SE | t | MLLM Avg |
|---|---:|---:|---:|---:|
| siglip2_l16_384 | +0.00516 | 0.00161 | **+3.19** | 53.01 |
| pe_lang_g14_448 | +0.00128 | 0.00189 | +0.67 | 58.03 |
| dino_vitb16 | −0.00350 | 0.00148 | **−2.37** | 34.84 |

跨 encoder 分布宽度约 0.009，SE 0.0017，信噪比 ~5。**64 条 query 时 SE≈0.0037，比效应量还大**——
早期用 64 query 得到的排序（含"pe pair 翻对了"）是噪声，pe_lang 在三次运行里给出
+0.00345 / −0.00218 / +0.00128。要分辨接近的 encoder 需要 1000+ 条 query。

### 判决标准

跑完 26 个后：一族抽一个 Spearman > 0.5，且 dino/webssl_mae 族显著偏低 -> 继续；否则关掉。
注意 pe_lang（全表 MLLM 最好）目前 t=0.67，如果扩量后它仍然分不开 0，这个指标就选不出 top-1。
