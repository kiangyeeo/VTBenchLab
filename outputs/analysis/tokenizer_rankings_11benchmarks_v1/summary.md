# Tokenizer ranking across 11 evaluations

Each cell is `rank (score %)`. Rank 1 is best. DTD uses a tied rank 4 for UniTok and TokLIP-S because their stored accuracies are exactly equal.

| Tokenizer | ImageNet-1K 2-shot | VOC2007 multi-label | CIFAR100 | Food101 | OxfordPets | Flowers102 | StanfordCars | FGVCAircraft | DTD | SUN397 | Caltech101 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UniTok | 4 (52.39) | 4 (88.76) | 1 (86.48) | 2 (92.81) | 4 (93.08) | 4 (97.27) | 4 (92.15) | 5 (52.36) | 4= (81.01) | 3 (82.63) | 2 (95.93) |
| TokLIP-S | 2 (59.21) | 2 (90.32) | 4 (84.35) | 5 (89.40) | 3 (93.32) | 3 (98.21) | 3 (93.41) | 4 (62.86) | 4= (81.01) | 5 (80.67) | 5 (93.18) |
| TokLIP-L | 1 (62.49) | 1 (91.85) | 5 (82.36) | 3 (92.70) | 1 (94.60) | 2 (98.88) | 1 (94.01) | 1 (67.54) | 2 (82.71) | 2 (82.69) | 4 (93.50) |
| VILA-U | 3 (52.81) | 5 (88.73) | 2 (86.39) | 1 (94.23) | 2 (94.41) | 1 (99.27) | 2 (93.46) | 2 (63.85) | 1 (83.35) | 1 (83.41) | 3 (95.71) |
| MetaCLIP | 5 (45.86) | 3 (90.04) | 3 (86.31) | 4 (91.74) | 5 (93.02) | 5 (97.19) | 5 (92.14) | 3 (62.89) | 3 (81.33) | 4 (82.47) | 1 (97.77) |

## Metric and source conventions

- ImageNet-1K uses the three-support-seed mean Top-1 for 2-shot. This gives VILA-U (52.81) above UniTok (52.39), as requested.
- VOC2007 uses official test 11-point mAP from `outputs/voc2007_multilabel_linear_kornblith_v1/summary.csv`.
- The other nine datasets use Top-1 accuracy from the final JSON record in each `outputs/vae_linear_probing/<dataset>/<tokenizer>/results_eval_linear.json` file.
- Flowers102 and SUN397 contain validation followed by test records; the table uses the final test record. Scores are percentages and ranks are computed independently within each column.
