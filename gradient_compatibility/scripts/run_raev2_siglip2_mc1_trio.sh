#!/usr/bin/env bash
set -euo pipefail

cd /cache/ma-user/VTBenchLab

CONFIG="${CONFIG:-gradient_compatibility/configs/raev2_siglip2_mc1_trio_pilot.json}"
TOKENIZERS="${TOKENIZERS:-all}"
SEEDS="${SEEDS:-0}"
STAGES="${STAGES:-manifest,tokens,warmup,probe,summary}"

if [[ ",${STAGES}," == *",manifest,"* ]]; then
  python -m gradient_compatibility.data --config "${CONFIG}"
fi

if [[ ",${STAGES}," == *",tokens,"* ]]; then
  python -m gradient_compatibility.token_cache \
    --config "${CONFIG}" --tokenizers ${TOKENIZERS}
fi

if [[ ",${STAGES}," == *",warmup,"* ]]; then
  python -m gradient_compatibility.train_projector \
    --config "${CONFIG}" --tokenizers ${TOKENIZERS} --seeds ${SEEDS}
fi

if [[ ",${STAGES}," == *",probe,"* ]]; then
  python -m gradient_compatibility.probe \
    --config "${CONFIG}" --tokenizers ${TOKENIZERS} --seeds ${SEEDS}
fi

if [[ ",${STAGES}," == *",summary,"* ]]; then
  python -m gradient_compatibility.summarize \
    --config "${CONFIG}" --tokenizers ${TOKENIZERS} --seeds ${SEEDS}
fi
