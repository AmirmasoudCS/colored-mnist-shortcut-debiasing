## Table 1: baseline accuracy (rho 0.99, seed 2)
| Set | rho | Accuracy (%) | Description |
|---|---|---|---|
| train | 0.99 | 99.93 | training |
| val | 0.99 | 99.43 | validation |
| test_id | 0.99 | 99.37 | ID test |
| test_ood1 | 0.5 | 93.07 | OOD-1: unbiased |
| test_ood2 | 0.1 | 87.91 | OOD-2: reversed |

## Aligned vs conflicting samples
| Set | Aligned acc (%) | Conflicting acc (%) | Loss | n conflicting |
|---|---|---|---|---|
| train | 99.94 | 99.07 | 0.0019 | 538 |
| val | 99.54 | 89.00 | 0.0387 | 100 |
| test_id | 99.55 | 81.05 | 0.0332 | 95 |
| test_ood1 | 99.66 | 86.43 | 0.4203 | 4983 |
| test_ood2 | 99.70 | 86.59 | 0.7521 | 8996 |

## Accuracy per group (colour/digits)
| Set | red/0-4 | red/5-9 | green/0-4 | green/5-9 |
|---|---|---|---|---|
| test_id | 99.71 | 81.63 | 80.43 | 99.38 |
| test_ood1 | 99.84 | 84.18 | 88.53 | 99.47 |
| test_ood2 | 99.61 | 84.28 | 88.79 | 99.79 |
