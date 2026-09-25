"""SAIL alignment probing on VTBench's cached tokenizer features.

SAIL (CVPR 2025, `SAIL/`) freezes both towers and trains only a lightweight alignment layer
under a SigLIP loss; how well that layer can be made to align the two spaces is its measure
of vision-language alignment. `SAIL/scripts/alignment_probing.sh` runs this on CC3M-scale
precomputed embeddings, which this repository does not have.

This adapter keeps SAIL's model, loss and retrieval metrics unchanged and feeds them the
embeddings we already cache: `lar/features/<name>__coco4618.npy` on the vision side and
`lar/text/caption__coco4618.npy` (Qwen2.5-1.5B, the MLLM evaluation's own LLM family) on the
text side, both indexed by the same 4618 COCO ids. The train/val/test split is the one every
other readout in `outputs/analysis/` uses, so the alignment score is comparable with them.

It deliberately bypasses `SAIL/main.py`'s distributed / wandb / checkpoint-resume harness:
84 short single-GPU runs need none of it, and the metrics there are only logged, never saved.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

R = Path('/cache/ma-user/VTBenchLab')
sys.path.insert(0, str(R / 'SAIL'))
from model.sail_model import AlignmentLayer          # noqa: E402
from model.loss import SigLipLoss                    # noqa: E402
from train.train import get_siglip_metrics           # noqa: E402

OUT = R / 'outputs/analysis/alignment_probing'
# alignment_probing.sh's settings, minus the batch size, which cannot exceed our 2400 pairs.
TARGET_DIM, LINEAR_TYPE, LOGIT_SCALE, LOGIT_BIAS = 2048, 'linear', 20.0, -10.0
WEIGHT_DECAY, BETAS = 1e-7, (0.9, 0.99)
# SAIL uses one lr at batch 32768; at batch 512 the right lr is unknown, so it is swept and
# selected on validation, the way every other readout in this project selects its regulariser.
LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
BATCH, EPOCHS, EVAL_EVERY = 512, 200, 20


def score_split(model, image, text, device):
    model.eval()
    with torch.inference_mode():
        out = model(image.to(device), text.to(device))
        metrics = get_siglip_metrics(
            image_features=out['image_features'], text_features=out['text_features'],
            logit_scale=out['logit_scale'], logit_bias=out['logit_bias'],
        )
    model.train()
    return {k: float(v) for k, v in metrics.items()}


def probe(vision: np.ndarray, text: np.ndarray, split, device: str) -> dict:
    tr, va, te = split
    Xtr = torch.as_tensor(vision[tr], dtype=torch.float32, device=device)
    Ttr = torch.as_tensor(text[tr], dtype=torch.float32, device=device)
    Xva, Tva = torch.as_tensor(vision[va], dtype=torch.float32), torch.as_tensor(text[va], dtype=torch.float32)
    Xte, Tte = torch.as_tensor(vision[te], dtype=torch.float32), torch.as_tensor(text[te], dtype=torch.float32)
    loss_fn = SigLipLoss()
    steps = max(1, len(tr) // BATCH)
    best = (-1.0, None, None)
    for lr in LEARNING_RATES:
        torch.manual_seed(0)
        model = AlignmentLayer(
            vision_dimesion=vision.shape[1], text_dimension=text.shape[1],
            target_dimension=TARGET_DIM, linear_type=LINEAR_TYPE,
            logit_scale=LOGIT_SCALE, logit_bias=LOGIT_BIAS,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY, betas=BETAS)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * steps)
        generator = torch.Generator().manual_seed(0)
        for epoch in range(EPOCHS):
            order = torch.randperm(len(tr), generator=generator).to(device)
            for step in range(steps):
                idx = order[step * BATCH:(step + 1) * BATCH]
                out = model(Xtr[idx], Ttr[idx])
                optimizer.zero_grad(set_to_none=True)
                loss_fn(**out, output_dict=True)['contrastive_loss'].backward()
                optimizer.step()
                schedule.step()
            if (epoch + 1) % EVAL_EVERY == 0:
                val = score_split(model, Xva, Tva, device)
                selector = 0.5 * (val['image_to_text_R@1'] + val['text_to_image_R@1'])
                if selector > best[0]:
                    best = (selector, score_split(model, Xte, Tte, device), (lr, epoch + 1))
    result = {f'test_{k}': v for k, v in best[1].items()}
    result['val_selector'] = best[0]
    result['best_lr'], result['best_epoch'] = best[2]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='*')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text = np.load(R / 'lar/text/caption__coco4618.npy').astype(np.float32)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(text))
    split = (perm[:2400], perm[2400:3400], perm[3400:])
    names = sorted(
        p.name.split('__coco4618.npy')[0]
        for p in (R / 'lar/features').glob('*__coco4618.npy')
        if '.limit' not in p.name
    )
    if args.models:
        names = [n for n in names if n in args.models]
    path = OUT / 'alignment_scores.csv'
    rows = []
    if path.is_file():
        done = pd.read_csv(path)
        rows = done.to_dict('records')
        names = [n for n in names if n not in set(done['name'])]
        print(f'resuming: {len(rows)} done, {len(names)} to go')
    for k, name in enumerate(names):
        t0 = time.time()
        vision = np.load(R / f'lar/features/{name}__coco4618.npy').astype(np.float32)
        row = {'name': name, 'd': vision.shape[1], **probe(vision, text, split, device)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        print('%3d/%d %-32s I2T R@1=%.3f T2I R@1=%.3f mean_rank=%.1f lr=%.0e ep=%d (%.0fs)'
              % (k + 1, len(names), name, row['test_image_to_text_R@1'],
                 row['test_text_to_image_R@1'], row['test_image_to_text_mean_rank'],
                 row['best_lr'], row['best_epoch'], time.time() - t0))
        sys.stdout.flush()
    print('wrote', path)


if __name__ == '__main__':
    main()
