"""Is the multi-label gain independent of the readout, or only a ridge-conditional effect?

`p1_probe.py` crossed readout x standardization on one label set (COCO 80-way multi-hot).
The label contrast that motivated the Readout Battery -- `mAP_dom` (one-hot on the largest
object) versus `mAP_obj80` (80-way multi-hot) -- has only ever been run under the closed-form
ridge. This script runs the same one-hot labels through all four readout configurations, so
the label axis can be read at fixed readout and the readout axis at fixed labels.

Everything else is identical to p1_probe.py: same cached features, same 2400/1000/1218 split,
same alpha and learning-rate grids, same selection on validation mAP.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from p1_probe import R, OUT, ridge, sgd

def build_label_sets():
    ids = [int(line.strip()) for line in (R / 'lar/features/dinov2_large__coco4618.ids.txt').read_text().split()]
    position = {v: i for i, v in enumerate(ids)}
    inst = json.loads((R / 'data/gvt/raw/coco/annotations/instances_val2017.json').read_text())
    cats = sorted({c['id'] for c in inst['categories']})
    cmap = {c: i for i, c in enumerate(cats)}
    obj = np.zeros((len(ids), len(cats)), dtype=np.float32)
    area = np.zeros_like(obj)
    for a in inst['annotations']:
        if a['image_id'] in position:
            obj[position[a['image_id']], cmap[a['category_id']]] = 1
            area[position[a['image_id']], cmap[a['category_id']]] += a['area']
    dom = np.zeros_like(obj)                       # a7_coco.py's single-label control
    for i in range(len(ids)):
        if area[i].max() > 0:
            dom[i, area[i].argmax()] = 1
    return obj, dom


def main() -> None:
    only = sys.argv[1:] or None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Yobj, Ydom = build_label_sets()
    print('positives/image: obj80=%.2f dom=%.2f' % (Yobj.sum(1).mean(), Ydom.sum(1).mean()))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Yobj))
    split = (perm[:2400], perm[2400:3400], perm[3400:])
    names = sorted(
        p.name.split('__coco4618.npy')[0]
        for p in (R / 'lar/features').glob('*__coco4618.npy')
        if '.limit' not in p.name
    )
    if only:
        names = [n for n in names if n in only]
    path = OUT / 'readout_labels.csv'
    rows = []
    if path.is_file():                       # resume: this sweep is long enough to get killed
        done = pd.read_csv(path)
        rows = done.to_dict('records')
        finished = set(done['name'])
        names = [n for n in names if n not in finished]
        print('resuming: %d already done, %d to go' % (len(finished), len(names)))
    for k, name in enumerate(names):
        t0 = time.time()
        Z = np.load(R / f'lar/features/{name}__coco4618.npy').astype(np.float64)
        raw = Z - Z.mean(0)
        std = raw / (Z.std(0) + 1e-8)
        row = {'name': name}
        for label_tag, Y in (('obj80', Yobj), ('dom', Ydom)):
            row[f'ridge_std_{label_tag}'], _ = ridge(std, Y, split)
            row[f'ridge_raw_{label_tag}'], _ = ridge(raw, Y, split)
            row[f'sgd_std_{label_tag}'], _, _ = sgd(std, Y, split, device)
            row[f'sgd_raw_{label_tag}'], _, _ = sgd(raw, Y, split, device)
        rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        print('%3d/%d %-32s ridge_std obj=%.3f dom=%.3f | sgd_std obj=%.3f dom=%.3f | sgd_raw obj=%.3f dom=%.3f (%.0fs)'
              % (k + 1, len(names), name, row['ridge_std_obj80'], row['ridge_std_dom'],
                 row['sgd_std_obj80'], row['sgd_std_dom'], row['sgd_raw_obj80'], row['sgd_raw_dom'],
                 time.time() - t0))
        sys.stdout.flush()
    print('wrote', path)


if __name__ == '__main__':
    main()
