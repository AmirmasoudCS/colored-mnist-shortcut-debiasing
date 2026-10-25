## Colour-flipped digits (rho 0.99, seed 0)
1416 of 9964 naturally-correct digits (14.21%) are classified wrongly once their colour is reversed.

| Direction | Flipped | Naturally correct | Flip rate (%) |
|---|---|---|---|
| digits 0-4 painted green | 916 | 5130 | 17.86 |
| digits 5-9 painted red | 500 | 4834 | 10.34 |

Wrong answer crosses the 0-4 / 5-9 boundary (follows the colour): 97.7%.

Control: among the 195 errors this model makes on colour-free (grayscale) images, 70.8% cross the boundary (look-alike confusions such as 4/9 or 3/5 straddle it too).

Top confusions (true -> predicted): 4->9 (264), 2->7 (120), 3->8 (115), 3->5 (98), 2->8 (78)

| Map statistics over 1000 flipped digits | Grad-CAM mass on digit | Gradient mass on digit | Gradient share in own colour channel |
|---|---|---|---|
| natural colour, explaining true class | 0.325 | 0.386 | 0.537 |
| reversed colour, explaining true class | 0.251 | 0.369 | 0.540 |
| reversed colour, explaining wrong class | 0.272 | 0.382 | 0.548 |

Digit pixels cover 0.170 of the image (uniform-spread reference).
