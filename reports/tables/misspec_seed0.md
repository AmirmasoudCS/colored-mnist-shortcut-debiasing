## Misspecification tests (rho 0.9, seed 0, n = 1000; 95% CI at most +-3.1 pts)

**Accuracy (%)**
| Shape | natural | reversed | grayscale |
|---|---|---|---|
| clean | 99.6 | 97.9 | 99.1 |
| rot90 | 25.1 | 16.4 | 21.7 |
| occlude_right_half | 80.2 | 66.8 | 74.5 |

**Prediction follows the colour (%)** (about 50 = colour has no pull)
| Shape | natural | reversed |
|---|---|---|
| clean | 99.8 | 2.0 |
| rot90 | 66.6 | 59.6 |
| occlude_right_half | 89.4 | 28.3 |
