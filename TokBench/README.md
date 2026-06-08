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

**3. Image Reconstruct  → evaluate → aggregate**

```bash
cd TokBench/tokenzier_vae_scripts/image_scripts
PADDING_SIZES="256 512 1024" bash resize.sh

cd TokBench
RES=256 bash image_eval.sh
RES=512 bash image_eval.sh
RES=1024 bash image_eval.sh
```

**4. Video Reconstruct  → evaluate → aggregate**

```bash
cd TokBench/tokenzier_vae_scripts/video_scripts
SHORT_SIZES="256 480" bash resize.sh

cd TokBench
RES=256 bash video_eval.sh
RES=480 bash video_eval.sh
```
