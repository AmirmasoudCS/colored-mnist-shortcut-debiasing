## Misspecification tests (rho 0.99, seed 0, n = 1000; 95% CI at most +-3.1 pts)

**Accuracy (%)**
| Shape | natural | reversed | grayscale |
|---|---|---|---|
| clean | 99.9 | 85.7 | 97.9 |
| rot90 | 25.8 | 4.0 | 16.2 |
| occlude_right_half | 83.1 | 48.9 | 69.9 |

**Prediction follows the colour (%)** (about 50 = colour has no pull)
| Shape | natural | reversed |
|---|---|---|
| clean | 99.9 | 14.1 |
| rot90 | 89.1 | 85.0 |
| occlude_right_half | 96.0 | 46.7 |
