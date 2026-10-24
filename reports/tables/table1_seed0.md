## Table 1: baseline accuracy
| Set | rho | Accuracy (%) | Description |
|---|---|---|---|
| train | 0.9 | 99.97 | training |
| val | 0.9 | 99.33 | validation |
| test_id | 0.9 | 99.46 | ID test |
| test_ood1 | 0.5 | 98.65 | OOD-1: unbiased |
| test_ood2 | 0.1 | 98.19 | OOD-2: reversed |

## Aligned vs conflicting samples
| Set | Aligned acc (%) | Conflicting acc (%) | Loss |
|---|---|---|---|
| train | 99.97 | 99.94 | 0.0011 |
| val | 99.49 | 97.88 | 0.0310 |
| test_id | 99.55 | 98.64 | 0.0217 |
| test_ood1 | 99.60 | 97.69 | 0.0655 |
| test_ood2 | 99.90 | 98.00 | 0.0957 |

## Accuracy per group (colour/digits)
| Set | red/0-4 | red/5-9 | green/0-4 | green/5-9 |
|---|---|---|---|---|
| test_id | 99.74 | 98.41 | 98.86 | 99.36 |
| test_ood1 | 99.84 | 97.50 | 97.87 | 99.35 |
| test_ood2 | 99.81 | 97.71 | 98.27 | 100.00 |
