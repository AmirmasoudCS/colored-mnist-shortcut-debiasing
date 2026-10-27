## Table 1: baseline accuracy (rho 0.99, seed 1)
| Set | rho | Accuracy (%) | Description |
|---|---|---|---|
| train | 0.99 | 99.91 | training |
| val | 0.99 | 99.30 | validation |
| test_id | 0.99 | 99.41 | ID test |
| test_ood1 | 0.5 | 91.68 | OOD-1: unbiased |
| test_ood2 | 0.1 | 85.95 | OOD-2: reversed |

## Aligned vs conflicting samples
| Set | Aligned acc (%) | Conflicting acc (%) | Loss | n conflicting |
|---|---|---|---|---|
| train | 99.94 | 97.96 | 0.0026 | 538 |
| val | 99.45 | 84.00 | 0.0444 | 100 |
| test_id | 99.60 | 80.00 | 0.0343 | 95 |
| test_ood1 | 99.70 | 83.60 | 0.5101 | 4983 |
| test_ood2 | 99.70 | 84.42 | 0.8870 | 8996 |

## Accuracy per group (colour/digits)
| Set | red/0-4 | red/5-9 | green/0-4 | green/5-9 |
|---|---|---|---|---|
| test_id | 99.82 | 75.51 | 84.78 | 99.36 |
| test_ood1 | 99.92 | 82.47 | 84.66 | 99.47 |
| test_ood2 | 99.81 | 83.36 | 85.41 | 99.59 |
