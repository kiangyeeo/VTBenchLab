# Quick Start: TokBench
**1. Environment Setup**

```bash
conda activate TokBench
conda install -c pytorch -c nvidia pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8 -y
conda install "mkl<2025" -y

cd TokBench
pip install -i https://pypi.org/simple -r requirements.txt
pip install -e . --no-deps

export USE_TORCH=1
export LD_LIBRARY_PATH="$(python -c 'import os,glob,nvidia; print(":".join(glob.glob(os.path.join(os.path.dirname(nvidia.__file__),"*","lib"))))'):$LD_LIBRARY_PATH"
```

**2. Download the Benchmark Dataset** 

```bash
bash download_data.sh   #image
WITH_VIDEO=1 bash download_data.sh   #video
```

**3. (resize baseline) Image Reconstruct  → evaluate → aggregate** 

```bash
cd TokBench/tokenzier_vae_scripts/image_scripts
PADDING_SIZES="256 512 1024" bash resize.sh

cd TokBench
RES=256 bash image_eval.sh
RES=512 bash image_eval.sh
RES=1024 bash image_eval.sh
```

**4. (resize baseline) Video Reconstruct  → evaluate → aggregate**

```bash
cd TokBench/tokenzier_vae_scripts/video_scripts
SHORT_SIZES="256 480" bash resize.sh

cd TokBench
RES=256 bash video_eval.sh
RES=480 bash video_eval.sh
```

**Run other tokenizers**

```bash
cd TokBench/tokenzier_vae_scripts/image_scripts
PADDING_SIZES="256 512 1024" bash sdxl.sh

cd TokBench
MODEL_NAME=sdxl RES=256 OUT_DIR=image_outputs/sdxl_256 bash image_eval.sh
MODEL_NAME=sdxl RES=512 OUT_DIR=image_outputs/sdxl_512 bash image_eval.sh
MODEL_NAME=sdxl RES=1024 OUT_DIR=image_outputs/sdxl_1024 bash image_eval.sh
```

```bash
cd TokBench/tokenzier_vae_scripts/video_scripts
git clone --depth 1 https://github.com/Tencent-Hunyuan/HunyuanVideo.git
SHORT_SIZES="256 480" bash hunyuan.sh

cd TokBench
MODEL_NAME=hunyuan RES=256 bash video_eval.sh
MODEL_NAME=hunyuan RES=480 bash video_eval.sh
```

