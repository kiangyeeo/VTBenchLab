| Method                                 | Architecture    | Token type   | Link                                                         |
| -------------------------------------- | --------------- | ------------ | ------------------------------------------------------------ |
| GVTBench                               | Encoder-only    | Continuous   | https://github.com/TencentARC/GVT                            |
| CV-Bench                               | Encoder-only    | Continuous   | https://github.com/cambrian-mllm/cambrian                    |
| Zero-shot Classification and Retrieval | Encoder-only    | Continuous*  | https://github.com/LAION-AI/CLIP_benchmark                   |
| k-NN                                   | Encoder-only    | Both         | https://github.com/facebookresearch/dinov2                   |
| Linear Probing                         | Encoder-only    | Both         | https://github.com/facebookresearch/dinov2                   |
| SAIL (VL Alignment Probing)            | Encoder-only    | Both         | https://lezhang7.github.io/sail.github.io                    |
| freeze-align                           | Encoder-only    | Both         | https://github.com/mayug/freeze-align                        |
| FeatUp (dense probing)                 | Encoder-only    | Both         | https://github.com/mhamilton723/FeatUp                       |
| Law of Vision Rep. (AC)                | Encoder-only    | Both         | https://github.com/bronyayang/Law_of_Vision_Representation_in_MLLMs |
| OLA-VLM / VisPer-LM                    | Encoder-only    | Both         | https://praeclarumjj3.github.io/ola_vlm                      |
| TokBench                               | Encoder-Decoder | Both         | https://github.com/wjf5203/TokBench                          |
| VTBench                                | Encoder-Decoder | Both†        | https://github.com/huawei-lin/VTBench                        |
| TokenBench                             | Encoder-Decoder | Both (video) | https://github.com/NVlabs/TokenBench                         |
| swiss-ai/benchmark-image-tokenzier     | Encoder-Decoder | Discrete     | https://github.com/swiss-ai/benchmark-image-tokenzier        |
| GenEval                                | Encoder-Decoder | Both         | https://github.com/djghosh13/geneval                         |
| AR Probing (GigaTok)                   | Encoder-Decoder | Discrete     | https://arxiv.org/abs/2504.08736                             |
| UniEval                                | Unified         | Both         | https://github.com/xmed-lab/UniEval                          |
| MME-Unify                              | Unified         | Both         | https://github.com/MME-Benchmarks/MME-Unify                  |
| InterleavedBench                       | Interleaved     | Both         | https://arxiv.org/abs/2406.14643                             |
| ISG-Bench                              | Interleaved     | Both         | https://github.com/Dongping-Chen/ISG                         |
| OpenING                                | Interleaved     | Both         | https://opening-benchmark.github.io                          |
| MMIE                                   | Interleaved     | Both         | https://arxiv.org/abs/2410.10139                             |

```bash
# 0. fresh env
conda create -n dino python=3.10 -y && conda activate dino
cd /cache/ma-user/VTBenchLab/dinov2
pip install -r requirements.txt
pip install -e .

# 1. backbone (~1.1 GB)
bash knn_tools/download_model.sh

# 2. ImageNet-1k via HF mirror (~140 GB download + ~140 GB output)
export HF_ENDPOINT=https://hf-mirror.com                 
python knn_tools/prepare_imagenet.py --out /cache/ma-user/VTBenchLab/data/imagenet1k

# 3. metadata
PYTHONPATH=. python knn_tools/dump_extra.py \
    --root  /cache/ma-user/VTBenchLab/data/imagenet1k \
    --extra /cache/ma-user/VTBenchLab/data/imagenet1k/extra

# 4. run k-NN
bash knn_tools/run_knn.sh


cd /cache/ma-user/VTBenchLab/dinov2
for m in vits14 vitb14 vitg14; do bash knn_tools/download_model.sh "$m"; done
for m in vits14 vitb14 vitg14; do bash knn_tools/run_knn.sh "$m"; done




# Run DINO 

cd /cache/ma-user/VTBenchLab/dinov2
export HF_ENDPOINT=https://hf-mirror.com
DATA=/cache/ma-user/VTBenchLab/data/imagenet1k
OUT=/cache/ma-user/VTBenchLab/outputs

run () {  # run <timm_tag> <out_subdir> [extra args...]
  PYTHONPATH=. torchrun --nproc_per_node=1 knn_tools/run_knn_timm.py \
    --model "$1" --output-dir "$OUT/$2" \
    --train-dataset "ImageNet:split=TRAIN:root=$DATA:extra=$DATA/extra" \
    --val-dataset   "ImageNet:split=VAL:root=$DATA:extra=$DATA/extra" "${@:3}"
}
run vit_base_patch8_224.dino  knn_dino_vitb8
run vit_base_patch16_224.dino   knn_dino_vitb16
run vit_small_patch8_224.dino  knn_dino_vits8



cd /cache/ma-user/VTBenchLab/dinov2
export HF_ENDPOINT=https://hf-mirror.com
DATA=/cache/ma-user/VTBenchLab/data/imagenet1k

PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=29600 knn_tools/run_knn_timm.py \
    --model vit_small_patch8_224.dino \
    --output-dir /cache/ma-user/VTBenchLab/outputs/knn_dino_vits8 \
    --train-dataset "ImageNet:split=TRAIN:root=$DATA:extra=$DATA/extra" \
    --val-dataset   "ImageNet:split=VAL:root=$DATA:extra=$DATA/extra" \
    --batch-size 512

cd /cache/ma-user/VTBenchLab/dinov2
DATA=/cache/ma-user/VTBenchLab/data/imagenet1k
WEIGHTS=/cache/ma-user/VTBenchLab/checkpoints/dinov2/dinov2_vitl14_pretrain.pth
CONFIG=dinov2/configs/eval/vitl14_pretrain.yaml
OUT=/cache/ma-user/VTBenchLab/outputs/dinov2_linear_vitl14

PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=29700 dinov2/eval/linear.py \
    --config-file $CONFIG \
    --pretrained-weights $WEIGHTS \
    --output-dir $OUT \
    --train-dataset "ImageNet:split=TRAIN:root=$DATA:extra=$DATA/extra" \
    --val-dataset   "ImageNet:split=VAL:root=$DATA:extra=$DATA/extra"



cd /cache/ma-user/VTBenchLab/dinov2
bash knn_tools/run_linear.sh vits14 --num-workers 16 lp2
bash knn_tools/run_linear.sh vitb14 --num-workers 16 lp3
bash knn_tools/run_linear.sh vitl14 --num-workers 16 lp
bash knn_tools/run_linear.sh vitg14 --num-workers 16 lp4
 

cd /cache/ma-user/VTBenchLab/dinov2

HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=. torchrun --standalone --nproc_per_node=1 knn_tools/run_linear_timm.py \
  --model vit_small_patch16_224.dino \
  --output-dir /cache/ma-user/VTBenchLab/outputs/linear_dino_vits16 \
  --train-dataset "ImageNet:split=TRAIN:root=/cache/ma-user/VTBenchLab/data/imagenet1k:extra=/cache/ma-user/VTBenchLab/data/imagenet1k/extra" \
  --val-dataset   "ImageNet:split=VAL:root=/cache/ma-user/VTBenchLab/data/imagenet1k:extra=/cache/ma-user/VTBenchLab/data/imagenet1k/extra" \
  --num-workers 4

HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=. torchrun --standalone --nproc_per_node=1 knn_tools/run_linear_timm.py \
  --model vit_small_patch8_224.dino \
  --output-dir /cache/ma-user/VTBenchLab/outputs/linear_dino_vits8 \
  --train-dataset "ImageNet:split=TRAIN:root=/cache/ma-user/VTBenchLab/data/imagenet1k:extra=/cache/ma-user/VTBenchLab/data/imagenet1k/extra" \
  --val-dataset   "ImageNet:split=VAL:root=/cache/ma-user/VTBenchLab/data/imagenet1k:extra=/cache/ma-user/VTBenchLab/data/imagenet1k/extra" \
  --num-workers 4

HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=. torchrun --standalone --nproc_per_node=1 knn_tools/run_linear_timm.py \
  --model vit_base_patch16_224.dino \
  --output-dir /cache/ma-user/VTBenchLab/outputs/linear_dino_vitb16 \
  --train-dataset "ImageNet:split=TRAIN:root=/cache/ma-user/VTBenchLab/data/imagenet1k:extra=/cache/ma-user/VTBenchLab/data/imagenet1k/extra" \
  --val-dataset   "ImageNet:split=VAL:root=/cache/ma-user/VTBenchLab/data/imagenet1k:extra=/cache/ma-user/VTBenchLab/data/imagenet1k/extra" \
  --num-workers 4

HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=. torchrun --standalone --nproc_per_node=1 knn_tools/run_linear_timm.py \
  --model vit_base_patch8_224.dino \
  --output-dir /cache/ma-user/VTBenchLab/outputs/linear_dino_vitb8 \
  --train-dataset "ImageNet:split=TRAIN:root=/cache/ma-user/VTBenchLab/data/imagenet1k:extra=/cache/ma-user/VTBenchLab/data/imagenet1k/extra" \
  --val-dataset   "ImageNet:split=VAL:root=/cache/ma-user/VTBenchLab/data/imagenet1k:extra=/cache/ma-user/VTBenchLab/data/imagenet1k/extra" \
  --num-workers 4


```

