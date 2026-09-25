# PASCAL VOC 2007 multi-label linear probing

Official VOC2007 11-point mAP (%). L2 lambda is selected on val; the final head is refit on train+val and evaluated on test.

| model | feature dim | selected lambda | val mAP | test mAP | convergence |
|---|---:|---:|---:|---:|---|
| unitok | 1024 | 100 | 89.21 | 88.76 | ok |
| vilau | 1024 | 17.7827941004 | 88.18 | 88.73 | ok |
| metaclip | 768 | 100 | 89.72 | 90.04 | ok |
| toklips | 1152 | 17.7827941004 | 89.67 | 90.32 | ok |
| toklipl | 1152 | 5.6234132519 | 91.99 | 91.85 | ok |
