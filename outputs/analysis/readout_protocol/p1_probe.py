"""Decompose the ridge-vs-SGD readout gap on IDENTICAL cached features.

`mAP_obj80` (a7_coco.py, closed-form ridge on standardized cached features) predicts MLLM
well; `coco_mAP` (COCO2014 SGD+13LR on a live augmented backbone) does not, on the same
models with the same 80-way multi-hot labels. Three things differ at once: the optimizer,
feature standardization, and data scale + augmentation.

This script holds data scale and augmentation fixed (the same 2400 cached COCO val2017
images a7 used) and crosses the other two axes, so the ridge-vs-SGD and standardized-vs-raw
contributions can be read off separately.
"""
import json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

R = Path('/cache/ma-user/VTBenchLab')
OUT = R / 'outputs/analysis/readout_protocol'
# The 13-point grid, batch scaling, schedule and head init are copied from the ImageNet
# probing protocol so the SGD arm is that protocol's readout, not a new one.
BASE_LRS = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.3, 0.5]
BATCH, EPOCHS, EVAL_EVERY = 256, 100, 10
ALPHAS = np.logspace(-2, 6, 17)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Column-wise AP, matching sklearn's step-interpolated definition."""
    order = np.argsort(-scores, axis=0)
    y = np.take_along_axis(y_true, order, axis=0)
    tp = np.cumsum(y, axis=0)
    precision = tp / np.arange(1, len(y) + 1)[:, None]
    positives = y.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (precision * y).sum(0) / positives


def mean_ap(y_true: np.ndarray, scores: np.ndarray, keep: np.ndarray) -> float:
    return float(np.nanmean(average_precision(y_true[:, keep], scores[:, keep])))


def build_labels():
    ids = [int(line.strip()) for line in (R / 'lar/features/dinov2_large__coco4618.ids.txt').read_text().split()]
    position = {v: i for i, v in enumerate(ids)}
    inst = json.loads((R / 'data/gvt/raw/coco/annotations/instances_val2017.json').read_text())
    cats = sorted({c['id'] for c in inst['categories']})
    cmap = {c: i for i, c in enumerate(cats)}
    Y = np.zeros((len(ids), len(cats)), dtype=np.float32)
    for a in inst['annotations']:
        if a['image_id'] in position:
            Y[position[a['image_id']], cmap[a['category_id']]] = 1
    return Y


def ridge(X, Y, split):
    tr, va, te = split
    U, S, Vt = np.linalg.svd(X[tr], full_matrices=False)
    UtY = U.T @ Y[tr]
    keep_va = Y[va].sum(0) >= 3
    best = (-9.0, None)
    for alpha in ALPHAS:
        W = Vt.T @ ((S / (S**2 + alpha))[:, None] * UtY)
        score = mean_ap(Y[va], X[va] @ W, keep_va)
        if score > best[0]:
            best = (score, W)
    keep_te = Y[te].sum(0) >= 3
    return mean_ap(Y[te], X[te] @ best[1], keep_te), best[0]


def sgd(X, Y, split, device):
    tr, va, te = split
    Xtr = torch.as_tensor(X[tr], dtype=torch.float32, device=device)
    Ytr = torch.as_tensor(Y[tr], dtype=torch.float32, device=device)
    Xva = torch.as_tensor(X[va], dtype=torch.float32, device=device)
    Xte = torch.as_tensor(X[te], dtype=torch.float32, device=device)
    keep_va, keep_te = Y[va].sum(0) >= 3, Y[te].sum(0) >= 3
    steps_per_epoch = max(1, len(tr) // BATCH)
    best = (-9.0, None, None)
    generator = torch.Generator(device='cpu').manual_seed(0)
    for base_lr in BASE_LRS:
        torch.manual_seed(0)
        head = torch.nn.Linear(X.shape[1], Y.shape[1]).to(device)
        torch.nn.init.normal_(head.weight, 0.0, 0.01)
        torch.nn.init.zeros_(head.bias)
        effective_lr = base_lr * BATCH / 256
        optimizer = torch.optim.SGD(head.parameters(), lr=effective_lr, momentum=0.9, weight_decay=0.0)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS * steps_per_epoch, eta_min=0.0
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for epoch in range(EPOCHS):
            perm = torch.randperm(len(tr), generator=generator).to(device)
            for step in range(steps_per_epoch):
                idx = perm[step * BATCH:(step + 1) * BATCH]
                optimizer.zero_grad(set_to_none=True)
                loss_fn(head(Xtr[idx]), Ytr[idx]).backward()
                optimizer.step()
                schedule.step()
            if (epoch + 1) % EVAL_EVERY == 0:
                with torch.no_grad():
                    score = mean_ap(Y[va], head(Xva).cpu().numpy(), keep_va)
                if score > best[0]:
                    with torch.no_grad():
                        best = (score, head(Xte).cpu().numpy(), base_lr)
    return mean_ap(Y[te], best[1], keep_te), best[0], best[2]


def main() -> None:
    only = sys.argv[1:] or None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Y = build_labels()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Y))
    split = (perm[:2400], perm[2400:3400], perm[3400:])
    names = sorted(
        p.name.split('__coco4618.npy')[0]
        for p in (R / 'lar/features').glob('*__coco4618.npy')
        if '.limit' not in p.name
    )
    if only:
        names = [n for n in names if n in only]
    rows, path = [], OUT / 'readout_2x2.csv'
    for k, name in enumerate(names):
        t0 = time.time()
        Z = np.load(R / f'lar/features/{name}__coco4618.npy').astype(np.float64)
        raw = Z - Z.mean(0)                       # centering only; no per-dim rescale
        std = raw / (Z.std(0) + 1e-8)             # a7_coco.py's preprocessing
        row = {'name': name, 'd': Z.shape[1]}
        row['ridge_std'], _ = ridge(std, Y, split)
        row['ridge_raw'], _ = ridge(raw, Y, split)
        row['sgd_std'], _, row['sgd_std_lr'] = sgd(std, Y, split, device)
        row['sgd_raw'], _, row['sgd_raw_lr'] = sgd(raw, Y, split, device)
        rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        print('%3d/%d %-34s ridge_std=%.3f ridge_raw=%.3f sgd_std=%.3f sgd_raw=%.3f  (%.0fs)'
              % (k + 1, len(names), name, row['ridge_std'], row['ridge_raw'],
                 row['sgd_std'], row['sgd_raw'], time.time() - t0))
        sys.stdout.flush()
    print('wrote', path)


if __name__ == '__main__':
    main()
