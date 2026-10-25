## Colour-flipped digits (rho 0.9, seed 0)
164 of 9958 naturally-correct digits (1.65%) are classified wrongly once their colour is reversed.

| Direction | Flipped | Naturally correct | Flip rate (%) |
|---|---|---|---|
| digits 0-4 painted green | 75 | 5126 | 1.46 |
| digits 5-9 painted red | 89 | 4832 | 1.84 |

Wrong answer crosses the 0-4 / 5-9 boundary (follows the colour): 92.1% (about 55.6% if the wrong class were random).

Top confusions (true -> predicted): 9->4 (17), 4->9 (16), 5->3 (13), 2->8 (10), 7->1 (9)

| Map statistics over 164 flipped digits | Grad-CAM mass on digit | Gradient mass on digit | Gradient share in own colour channel |
|---|---|---|---|
| natural colour, explaining true class | 0.262 | 0.359 | 0.541 |
| reversed colour, explaining true class | 0.270 | 0.358 | 0.532 |
| reversed colour, explaining wrong class | 0.286 | 0.361 | 0.539 |

Digit pixels cover 0.178 of the image (uniform-spread reference).
