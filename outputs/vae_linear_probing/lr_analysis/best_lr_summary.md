# Linear probing LR boundary summary

`configured LR` is the CLI grid value; `effective LR` includes DINOv2's `batch_size * world_size / 256` scaling.

| task | model | configured LR | effective LR | blocks | avgpool | val top-1 | boundary |
|---|---|---:|---:|---:|---|---:|---|
| caltech101 | metaclip | 0.01 | 0.005 | 4 | False | 97.77 | interior |
| caltech101 | toklipl | 0.05 | 0.025 | 1 | False | 93.50 | interior |
| caltech101 | toklips | 0.02 | 0.01 | 1 | True | 93.18 | interior |
| caltech101 | unitok | 0.01 | 0.005 | 4 | True | 95.93 | interior |
| caltech101 | vilau | 0.02 | 0.01 | 1 | False | 95.71 | interior |
| cifar100 | metaclip | 0.1 | 0.05 | 1 | True | 86.31 |  **high** |
| cifar100 | toklipl | 0.1 | 0.05 | 4 | False | 82.36 |  **high** |
| cifar100 | toklips | 0.05 | 0.025 | 4 | True | 84.35 | interior |
| cifar100 | unitok | 0.02 | 0.01 | 4 | False | 86.48 | interior |
| cifar100 | vilau | 0.1 | 0.05 | 1 | False | 86.39 |  **high** |
| dtd | metaclip | 0.005 | 0.0025 | 1 | True | 81.33 | interior |
| dtd | toklipl | 0.01 | 0.005 | 4 | False | 82.71 | interior |
| dtd | toklips | 0.01 | 0.005 | 1 | False | 81.01 | interior |
| dtd | unitok | 0.005 | 0.0025 | 4 | False | 81.01 | interior |
| dtd | vilau | 0.005 | 0.0025 | 4 | False | 83.35 | interior |
| fgvc_aircraft | metaclip | 0.1 | 0.05 | 1 | False | 62.89 |  **high** |
| fgvc_aircraft | toklipl | 0.1 | 0.05 | 4 | False | 67.54 |  **high** |
| fgvc_aircraft | toklips | 0.1 | 0.05 | 4 | True | 62.86 |  **high** |
| fgvc_aircraft | unitok | 0.1 | 0.05 | 1 | False | 52.36 |  **high** |
| fgvc_aircraft | vilau | 0.1 | 0.05 | 4 | True | 63.85 |  **high** |
| flowers102 | metaclip | 0.02 | 0.01 | 1 | True | 98.04 | interior |
| flowers102 | toklipl | 0.01 | 0.005 | 4 | False | 99.02 | interior |
| flowers102 | toklips | 0.1 | 0.05 | 4 | False | 98.63 |  **high** |
| flowers102 | unitok | 0.01 | 0.005 | 4 | False | 97.94 | interior |
| flowers102 | vilau | 0.02 | 0.01 | 1 | False | 99.41 | interior |
| food101 | metaclip | 0.05 | 0.025 | 1 | True | 91.74 | interior |
| food101 | toklipl | 0.05 | 0.025 | 4 | True | 92.70 | interior |
| food101 | toklips | 0.05 | 0.025 | 4 | True | 89.40 | interior |
| food101 | unitok | 0.02 | 0.01 | 1 | True | 92.81 | interior |
| food101 | vilau | 0.02 | 0.01 | 4 | False | 94.23 | interior |
| imagenet1k | metaclip_b16_2pt5b | 0.05 | 0.025 | 4 | True | 80.60 | interior |
| oxford_pets | metaclip | 0.02 | 0.01 | 1 | False | 93.02 | interior |
| oxford_pets | toklipl | 0.1 | 0.05 | 4 | False | 94.60 |  **high** |
| oxford_pets | toklips | 0.05 | 0.025 | 4 | True | 93.32 | interior |
| oxford_pets | unitok | 0.005 | 0.0025 | 1 | True | 93.08 | interior |
| oxford_pets | vilau | 0.01 | 0.005 | 4 | False | 94.41 | interior |
| stanford_cars | metaclip | 0.02 | 0.01 | 1 | False | 92.14 | interior |
| stanford_cars | toklipl | 0.1 | 0.05 | 4 | False | 94.01 |  **high** |
| stanford_cars | toklips | 0.1 | 0.05 | 4 | True | 93.41 |  **high** |
| stanford_cars | unitok | 0.005 | 0.0025 | 4 | False | 92.15 | interior |
| stanford_cars | vilau | 0.05 | 0.025 | 4 | False | 93.46 | interior |
| sun397 | metaclip | 0.02 | 0.01 | 4 | True | 82.86 | interior |
| sun397 | toklipl | 0.05 | 0.025 | 4 | True | 82.98 | interior |
| sun397 | toklips | 0.05 | 0.025 | 4 | True | 80.86 | interior |
| sun397 | unitok | 0.01 | 0.005 | 4 | True | 82.28 | interior |
| sun397 | vilau | 0.02 | 0.01 | 4 | False | 83.41 | interior |
| imagenet1k | toklip_l_semantic_384 | 0.1 | 0.05 | 4 | True | 80.75 |  **high** |
| imagenet1k | toklip_l_zq_384 | 0.1 | 0.05 | 4 | True | 1.34 |  **high** |
| imagenet1k | toklip_s_semantic_256 | 0.1 | 0.05 | 4 | True | 77.72 |  **high** |
| imagenet1k | toklip_s_zq_256 | 0.1 | 0.05 | 4 | True | 2.18 |  **high** |
| imagenet1k | unitok | 0.1 | 0.05 | 1 | True | 80.62 |  **high** |
| imagenet1k | vilau_7b_256_semantic_penultimate | 0.02 | 0.01 | 4 | False | 83.58 | interior |

Historical runs contain 48 evaluated heads rather than the intended 52 because the smallest two effective LRs shared the same 5-decimal classifier name. Missing rows are retained in the CSV files with `status=missing_name_collision`.
