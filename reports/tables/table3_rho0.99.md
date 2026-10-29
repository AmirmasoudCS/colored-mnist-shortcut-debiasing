## Table 3: baseline vs mitigation (rho 0.99); accuracy in %, mean ± std over seeds

| Method | Seeds | ID acc | OOD-1 acc | OOD-2 acc | Gap ID-OOD1 | Gap ID-OOD2 | Worst group (OOD-1) |
|---|---|---|---|---|---|---|---|
| Baseline | 3 | 99.42 ± 0.05 | 92.35 ± 0.70 | 86.93 ± 0.98 | 7.07 ± 0.72 | 12.48 ± 1.00 | 82.66 ± 1.44 |
| Reweighting | 3 | 98.95 ± 0.31 | 97.96 ± 0.42 | 97.26 ± 0.61 | 0.99 ± 0.46 | 1.69 ± 0.73 | 96.16 ± 0.89 |
| Group DRO | 3 | 99.44 ± 0.13 | 96.29 ± 0.39 | 93.80 ± 0.75 | 3.15 ± 0.35 | 5.64 ± 0.67 | 91.89 ± 0.49 |
| LfF | 3 | 98.94 ± 0.13 | 95.63 ± 0.22 | 93.02 ± 0.38 | 3.31 ± 0.35 | 5.92 ± 0.52 | 90.46 ± 1.83 |

## Individual runs (accuracy %)

| Method | Seed | ID | OOD-1 | OOD-2 | Worst group (OOD-1) |
|---|---|---|---|---|---|
| Baseline | 0 | 99.47 | 92.30 | 86.94 | 81.33 |
| Baseline | 1 | 99.41 | 91.68 | 85.95 | 82.47 |
| Baseline | 2 | 99.37 | 93.07 | 87.91 | 84.18 |
| Reweighting | 0 | 98.78 | 98.32 | 97.93 | 97.17 |
| Reweighting | 1 | 99.30 | 98.06 | 97.12 | 95.85 |
| Reweighting | 2 | 98.76 | 97.50 | 96.73 | 95.47 |
| Group DRO | 0 | 99.29 | 95.97 | 93.11 | 92.21 |
| Group DRO | 1 | 99.55 | 96.17 | 93.70 | 91.32 |
| Group DRO | 2 | 99.48 | 96.73 | 94.59 | 92.13 |
| LfF | 0 | 99.06 | 95.45 | 92.69 | 89.54 |
| LfF | 1 | 98.97 | 95.58 | 92.93 | 89.27 |
| LfF | 2 | 98.80 | 95.87 | 93.44 | 92.56 |

## Change vs baseline with the same seed (points)

- **Reweighting**: ID -0.47 (per seed: -0.69, -0.11, -0.61); OOD-1 +5.61 (per seed: +6.02, +6.38, +4.43); OOD-2 +10.33 (per seed: +10.99, +11.17, +8.82)
- **Group DRO**: ID +0.02 (per seed: -0.18, +0.14, +0.11); OOD-1 +3.94 (per seed: +3.67, +4.49, +3.66); OOD-2 +6.87 (per seed: +6.17, +7.75, +6.68)
- **LfF**: ID -0.47 (per seed: -0.41, -0.44, -0.57); OOD-1 +3.28 (per seed: +3.15, +3.90, +2.80); OOD-2 +6.09 (per seed: +5.75, +6.98, +5.53)
