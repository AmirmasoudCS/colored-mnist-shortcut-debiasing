## Table 1: baseline accuracy (rho 0.99, seed 0)
| Set | rho | Accuracy (%) | Description |
|---|---|---|---|
| train | 0.99 | 99.80 | training |
| val | 0.99 | 99.22 | validation |
| test_id | 0.99 | 99.47 | ID test |
| test_ood1 | 0.5 | 92.30 | OOD-1: unbiased |
| test_ood2 | 0.1 | 86.94 | OOD-2: reversed |

## Aligned vs conflicting samples
| Set | Aligned acc (%) | Conflicting acc (%) | Loss | n conflicting |
|---|---|---|---|---|
| train | 99.82 | 97.21 | 0.0056 | 538 |
| val | 99.32 | 89.00 | 0.0446 | 100 |
| test_id | 99.64 | 82.11 | 0.0286 | 95 |
| test_ood1 | 99.66 | 84.89 | 0.4353 | 4983 |
| test_ood2 | 99.70 | 85.52 | 0.7665 | 8996 |

## Accuracy per group (colour/digits)
| Set | red/0-4 | red/5-9 | green/0-4 | green/5-9 |
|---|---|---|---|---|
| test_id | 99.82 | 79.59 | 84.78 | 99.44 |
| test_ood1 | 99.88 | 88.72 | 81.33 | 99.43 |
| test_ood2 | 99.81 | 89.26 | 81.97 | 99.59 |
