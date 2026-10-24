# Step 3 Report: ID and OOD evaluation of the baseline (Task 2)

**Date:** 24 October 2026

## Results

Baseline CNN (seed 0, final-epoch weights). *Aligned* = colour equals the digit's natural colour
(0-4 red, 5-9 green); *conflicting* = opposite colour.

**Table 1. Accuracy (%)**

| Set | rho | Accuracy | Aligned | Conflicting | Loss |
|---|---|---|---|---|---|
| train | 0.9 | 99.97 | 99.97 | 99.94 | 0.0011 |
| val | 0.9 | 99.33 | 99.49 | 97.88 | 0.0310 |
| ID test | 0.9 | 99.46 | 99.55 | 98.64 | 0.0217 |
| OOD-1 | 0.5 | 98.65 | 99.60 | 97.69 | 0.0655 |
| OOD-2 | 0.1 | 98.19 | 99.90 | 98.00 | 0.0957 |

**Accuracy per group (%)**

| Set | red/0-4 (aligned) | red/5-9 (conflicting) | green/0-4 (conflicting) | green/5-9 (aligned) |
|---|---|---|---|---|
| ID test | 99.74 | 98.41 | 98.86 | 99.36 |
| OOD-1 | 99.84 | 97.50 | 97.87 | 99.35 |
| OOD-2 | 99.81 | 97.71 | 98.27 | 100.00 |

![Baseline accuracy by sample type and group, and loss, for all five sets](figures/table1_baseline_eval.png)

*Figure 1. (a) Accuracy for all / aligned / conflicting samples with 95% binomial confidence
intervals. (b) Accuracy of the four colour/digit groups on the test sets; filled markers = aligned
groups, hollow = conflicting groups. (c) Cross-entropy loss. Panels (a) and (b) use a truncated
y-axis (starts at 95%).*

## Findings

1. **No collapse under distribution shift.** Accuracy falls only from 99.46% (ID) to 98.65%
   (OOD-1) and 98.19% (OOD-2), a drop of 0.8 and 1.3 points. The handout's expected strong
   degradation is not observed.
2. **Mild but consistent colour reliance.** On every set except train, conflicting samples are
   classified worse than aligned ones (by 0.9 to 1.9 points on the test sets; 1.6 on validation).
   The errors are concentrated in the conflicting groups (red/5-9, green/0-4).
3. **Loss reveals more than accuracy.** Loss rises 4.4x from ID (0.022) to OOD-2 (0.096) while
   accuracy changes by about one point: the model is less confident when colour contradicts shape.
4. **The gap is a generalisation gap.** On the training set conflicting and aligned accuracy are
   equal (99.94% vs 99.97%), but on unseen data conflicting samples lag behind.

## Interpretation

- **Shape is learnable here.** About 5,000 training samples (10%) have misleading colour, so shape
  is the only way to fit them. Colour alone predicts the label for at most 90% of samples, while
  shape is near-perfect for a CNN on MNIST. Training therefore favours shape.
- **rho = 0.9 is a weak bias.** The handout's simplicity-bias argument (the easy cue is learned
  first) matters when conflicting samples are rare or training is short; with 10% conflicting
  samples and 20 epochs, the model has time to learn both cues.
- **Colour still helps on unseen data.** On aligned samples colour and shape agree, so colour adds
  information; the model uses it as a secondary cue, which explains the small, consistent gap.
- **No data leakage.** Train, validation and test digits are disjoint by construction. ID accuracy
  (99.46%) matches typical CNN accuracy on plain MNIST, i.e. the task itself is easy. The three
  test sets share the same 10,000 digits, so differences between them are caused by colour alone.

## Caveats

- Single seed; the ID conflicting subset is small (about 1,000 samples, CI of roughly +-0.7 points),
  so the ID gap alone is weak evidence. The OOD-1 gap (about 5,000 conflicting samples) is clearer.
- Group results with few samples are noisy (e.g. 100.00% for green/5-9 on OOD-2, about 500 samples).
- These results show the model is *affected* by colour, not how strongly it relies on it; that is
  measured directly by the counterfactual tests in Task 3.

## Consequences for the next steps

- Mitigation methods have under 2 points of headroom, so several seeds are needed to separate
  real gains from noise.
- Optional extension: repeat training with a stronger bias (e.g. rho = 0.99) to show the regime
  in which the handout's collapse does appear.