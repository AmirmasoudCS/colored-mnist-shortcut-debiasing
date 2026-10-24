## Table 2: counterfactual tests (rho 0.99, seed 0, n = 10000)
| Condition | Accuracy (%) | Change vs clean (pts) | Loss |
|---|---|---|---|
| clean test set | 99.47 | +0.00 | 0.0286 |
| all aligned (reference) | 99.64 | +0.17 | 0.0176 |
| Test A: colours reversed | 85.56 | -13.91 | 0.8448 |
| Test B: grayscale (mean) | 96.41 | -3.06 | 0.1263 |
| Test B': grayscale (max, control) | 98.05 | -1.42 | 0.1368 |

Colour swing = acc(all aligned) - acc(all reversed) = 14.08 pts (0 for a pure shape model)
