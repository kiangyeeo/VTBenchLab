# Tokenizer 配置命名说明

连续视觉 Encoder 的配置 id 主要由以下信息组成：

```
mc1_g14_224_2.5b
│    │    │   │
│    │    │   └── 预训练数据规模（在 2.5B 图文数据上训练）
│    │    └───────── 输入分辨率
│    └────────────── Vision Transformer 架构（规模 + patch size）
└────────────────── 模型系列
```

## 缩写与后缀说明

|缩写|含义|
|---|---|
|B|Base|
|L|Large|
|H|Huge|
|G|Giant|

后面的数字表示 patch size。

|Backbone|Patch Size|224 输入视觉 token 数|
|---|---|---|
|ViT\-B/32|32×32|49|
|ViT\-B/16|16×16|196|
|ViT\-L/14|14×14|256|
|ViT\-G/14|14×14|256|

**特殊后缀**

- QuickGELU：表示使用 QuickGELU 激活函数。

- mT5：表示文本侧采用 mT5 tokenizer 配置。

# 1\.Tokenizer

## 1）连续

### CLIP 系列

- clip\_openai：OpenAI 训练的权重

- clip\_meta：Meta 训练的权重

|配置 id|说明|来源 / 下载|
|---|---|---|
|clip\_openai\_\_l14|OpenAI CLIP ViT\-L/14|[huggingface\.co/openai/clip\-vit\-large\-patch14](https://huggingface.co/openai/clip-vit-large-patch14)|
|clip\_meta\_\_l14|MetaCLIP ViT\-L/14|同 MetaCLIP1 L/14 @ 2\.5B：[l14\_fullcc2\.5b\.pt](https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_fullcc2.5b.pt)<br>（仓库 [facebookresearch/metaclip](https://github.com/facebookresearch/metaclip)）|

### MetaCLIP 

### [facebookresearch/metaclip](https://github.com/facebookresearch/metaclip)

**mc1：MetaCLIP 第一代**

|配置 id|架构|分辨率|数据规模|官方文件名|下载链接|
|---|---|---|---|---|---|
|mc1\_b32\_224\_400m|ViT\-B/32|224|400M|b32\_400m\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_400m.pt)|
|mc1\_b16\_224\_400m|ViT\-B/16|224|400M|b16\_400m\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_400m.pt)|
|mc1\_l14\_224\_400m|ViT\-L/14|224|400M|l14\_400m\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_400m.pt)|
|mc1\_b32\_224\_2\.5b|ViT\-B/32|224|2\.5B|b32\_fullcc2\.5b\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_fullcc2.5b.pt)|
|mc1\_b16\_224\_2\.5b|ViT\-B/16|224|2\.5B|b16\_fullcc2\.5b\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_fullcc2.5b.pt)|
|mc1\_l14\_224\_2\.5b|ViT\-L/14|224|2\.5B|l14\_fullcc2\.5b\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_fullcc2.5b.pt)|
|mc1\_h14\_224\_2\.5b|ViT\-H/14|224|2\.5B|h14\_fullcc2\.5b\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/h14_fullcc2.5b.pt)|
|mc1\_g14\_224\_2\.5b|ViT\-bigG/14 \(v1\.1\)|224|2\.5B|G14\_fullcc2\.5b\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/G14_fullcc2.5b.pt)|
|mc1\_h14\_224\_v1\.2|ViT\-H/14 \(v1\.2 Altogether\)|224|35B|h14\_v1\.2\_altogether\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/h14_v1.2_altogether.pt)|

**mc2：MetaCLIP 第二代，多语言预训练**

**全量 / Worldwide**

|配置 id|官方 model 名|Res|官方文件名|下载链接|
|---|---|---|---|---|
|mc2\_h14\_378|ViT\-H\-14\-378\-worldwide|378|metaclip2\_h14\_378px\_worldwide\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_h14_378px_worldwide.pt)|
|mc2\_g14\_224|ViT\-bigG\-14\-worldwide|224|metaclip2\_bigG14\_224px\_worldwide\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_bigG14_224px_worldwide.pt)|
|mc2\_g14\_378|ViT\-bigG\-14\-378\-worldwide|378|metaclip2\_bigG14\_378px\_worldwide\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_bigG14_378px_worldwide.pt)|

**Distilled**

|配置 id|官方 model 名|Res|官方文件名|下载链接|
|---|---|---|---|---|
|mc2\_s16\_224|ViT\-S\-16\-worldwide|224|metaclip2\_s16\_224px\_worldwide\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_s16_224px_worldwide.pt)|
|mc2\_s16\_384|ViT\-S\-16\-384\-worldwide|384|metaclip2\_s16\_384px\_worldwide\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_s16_384px_worldwide.pt)|
|mc2\_s16\_224\_mt5|ViT\-S\-16\-mT5\-worldwide|224|metaclip2\_s16\_224px\_mt5\_worldwide\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_s16_224px_mt5_worldwide.pt)|
|mc2\_m16\_224|ViT\-M\-16\-worldwide|224|metaclip2\_m16\_224px\_worldwide\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_m16_224px_worldwide.pt)|
|mc2\_m16\_384|ViT\-M\-16\-384\-worldwide|384|metaclip2\_m16\_384px\_worldwide\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_m16_384px_worldwide.pt)|
|mc2\_m16\_224\_mt5|ViT\-M\-16\-mT5\-worldwide|224|metaclip2\_m16\_224px\_mt5\_worldwide\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_m16_224px_mt5_worldwide.pt)|
|mc2\_b32\_224|ViT\-B\-32\-worldwide|224|metaclip2\_b32\_224px\_worldwide\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_b32_224px_worldwide.pt)|
|mc2\_b32\_384|ViT\-B\-32\-384\-worldwide|384|metaclip2\_b32\_384px\_worldwide\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_b32_384px_worldwide.pt)|
|mc2\_b32\_224\_mt5|ViT\-B\-32\-mT5\-worldwide|224|metaclip2\_b32\_224px\_mt5\_worldwide\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_b32_224px_mt5_worldwide.pt)|
|mc2\_b16\_224|ViT\-B\-16\-worldwide|224|metaclip2\_b16\_224px\_worldwide\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_b16_224px_worldwide.pt)|
|mc2\_b16\_384|ViT\-B\-16\-384\-worldwide|384|metaclip2\_b16\_384px\_worldwide\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_b16_384px_worldwide.pt)|
|mc2\_l14\_224|ViT\-L\-14\-worldwide|224|metaclip2\_l14\_224px\_worldwide\.pt|[link](https://dl.fbaipublicfiles.com/MMPT/metaclip/metaclip2_l14_224px_worldwide.pt)|

### SigLIP2

[big\_vision SigLIP2 README](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md)

|配置 id|ViT|Res|官方文件名|下载链接|
|---|---|---|---|---|
|siglip2\_b32\_256|B/32|256|siglip2\_b32\_256\.npz|[link](https://storage.googleapis.com/big_vision/siglip2/siglip2_b32_256.npz)|
|siglip2\_b16\_224|B/16|224|siglip2\_b16\_224\.npz|[link](https://storage.googleapis.com/big_vision/siglip2/siglip2_b16_224.npz)|
|siglip2\_b16\_256|B/16|256|siglip2\_b16\_256\.npz|[link](https://storage.googleapis.com/big_vision/siglip2/siglip2_b16_256.npz)|
|siglip2\_b16\_384|B/16|384|siglip2\_b16\_384\.npz|[link](https://storage.googleapis.com/big_vision/siglip2/siglip2_b16_384.npz)|
|siglip2\_b16\_512|B/16|512|siglip2\_b16\_512\.npz|[link](https://storage.googleapis.com/big_vision/siglip2/siglip2_b16_512.npz)|
|siglip2\_l16\_256|L/16|256|siglip2\_l16\_256\.npz|[link](https://storage.googleapis.com/big_vision/siglip2/siglip2_l16_256.npz)|
|siglip2\_l16\_384|L/16|384|siglip2\_l16\_384\.npz|[link](https://storage.googleapis.com/big_vision/siglip2/siglip2_l16_384.npz)|
|siglip2\_l16\_512|L/16|512|siglip2\_l16\_512\.npz|[link](https://storage.googleapis.com/big_vision/siglip2/siglip2_l16_512.npz)|
|siglip2\_sm14\_224|So400m/14|224|siglip2\_so400m14\_224\.npz|[link](https://storage.googleapis.com/big_vision/siglip2/siglip2_so400m14_224.npz)|
|siglip2\_sm14\_384|So400m/14|384†|siglip2\_so400m14\_384\.npz|[link](https://storage.googleapis.com/big_vision/siglip2/siglip2_so400m14_384.npz)|
|siglip2\_sm16\_256|So400m/16|256|siglip2\_so400m16\_256\.npz|[link](https://storage.googleapis.com/big_vision/siglip2/siglip2_so400m16_256.npz)|
|siglip2\_sm16\_384|So400m/16|384|siglip2\_so400m16\_384\.npz|[link](https://storage.googleapis.com/big_vision/siglip2/siglip2_so400m16_384.npz)|
|siglip2\_sm16\_512|So400m/16|512|siglip2\_so400m16\_512\.npz|[link](https://storage.googleapis.com/big_vision/siglip2/siglip2_so400m16_512.npz)|
|siglip2\_g16\_256|g\-opt/16|256|siglip2\_g\-opt16\_256\.npz|[link](https://storage.googleapis.com/big_vision/siglip2/siglip2_g-opt16_256.npz)|
|siglip2\_g16\_384|g\-opt/16|384|siglip2\_g\-opt16\_384\.npz|[link](https://storage.googleapis.com/big_vision/siglip2/siglip2_g-opt16_384.npz)|

## 2）离散

|配置 id|类型|分辨率 / tokens|codebook|仓库|下载|
|---|---|---|---|---|---|
|unitok\_attn|UniTok|256 / 256|32768|[FoundationVision/UniTok](https://github.com/FoundationVision/UniTok)|[HF unitok\_tokenizer\.pth](https://huggingface.co/FoundationVision/unitok_tokenizer/blob/main/unitok_tokenizer.pth)|
|vilau\_256|VILA\-U|256 / 256|16384|[mit\-han\-lab/vila\-u](https://github.com/mit-han-lab/vila-u)|[HF mit\-han\-lab/vila\-u\-7b\-256](https://huggingface.co/mit-han-lab/vila-u-7b-256)（用其中 vision\_tower/）|
|toklip\_s\_256|TokLIP\-S|256|16384|[TencentARC/TokLIP](https://github.com/TencentARC/TokLIP)|[HF TokLIP\_S\_256\.pt](https://huggingface.co/TencentARC/TokLIP/blob/main/TokLIP_S_256.pt)|
|toklip\_l\_384|TokLIP\-L|384|16384|同上|[HF TokLIP\_L\_384\.pt](https://huggingface.co/TencentARC/TokLIP/blob/main/TokLIP_L_384.pt)|

> TokLIP 依赖 SigLIP2 backbone（ViT\-SO400M\-16\-SigLIP2\-\{256,384\}\-toklip）；VQGAN 相关权重见 TokLIP README（如 LlamaGen VQ）。
> 
> 



