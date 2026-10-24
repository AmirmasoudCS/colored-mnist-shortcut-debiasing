## Table 2: counterfactual tests (rho 0.9, seed 0, n = 10000)
| Condition | Accuracy (%) | Change vs clean (pts) | Loss |
|---|---|---|---|
| clean test set | 99.46 | +0.00 | 0.0217 |
| all aligned (reference) | 99.58 | +0.12 | 0.0164 |
| Test A: colours reversed | 98.02 | -1.44 | 0.1067 |
| Test B: grayscale (mean) | 98.80 | -0.66 | 0.0395 |
| Test B': grayscale (max, control) | 99.17 | -0.29 | 0.0712 |

Colour swing = acc(all aligned) - acc(all reversed) = 1.56 pts (0 for a pure shape model)
