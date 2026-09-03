set -u
source /home/ma-user/miniconda3/etc/profile.d/conda.sh
conda activate TokBench
export TOKENIZERS_PARALLELISM=false
cd /cache/ma-user/VTBenchLab
python mechanistic/addressability.py \
  --models pe_lang_g14_448 pe_lang_l14_448 pe_core_g14_448 pe_core_b16_224 \
           siglip2_g16_384 siglip2_l16_384 siglip2_b16_224 siglip2_b32_256 \
           mc2_l14_224 mc2_b32_224 mc1_l14_224_2.5b mc1_b32_224_400m clip_openai__l14 \
           dinov2_large dinov2_base dinov2_small dino_vitb16 dino_vitb8 dino_vits16 \
           webssl_mae300m_full2b_224 webssl_mae3b_full2b_224 pixio_vitl16 pixio_vitb16 \
           eupe_vit_b toklip_l_384 unitok_attn \
  --n-images 400 --queries 400 --draws 1 --distractors 7 --tokens-per-image 64 \
  --projector-root gradient_compatibility/artifacts/full_sweep_v1/projectors \
  --out mechanistic/out/addressability_full.json
