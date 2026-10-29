# Step 9 Report: Learning from Failure (LfF) and the final Table 3 (Task 4)

**Date:** 25 October 2026

## Method

LfF (Nam et al., 2020) is added to the methods of Step 8. It needs **no colour labels**. Two networks
are trained together on the same batches:

- a **biased model**, trained with the generalised cross-entropy loss (1 - p^q) / q (q = 0.7), whose
  gradient favours samples it already fits and so amplifies the easiest cue (colour);
- a **debiased model**, trained with cross-entropy weighted per sample by
  W = CE_biased / (CE_biased + CE_debiased). Samples that the biased model fails on, i.e. those where
  the shortcut does not hold, receive large weights.

Per-sample losses are exponential moving averages (factor 0.7) and are divided by the maximum of their digit class.
The debiased model is the one evaluated. Setup as before: rho = 0.99, SimpleCNN, Adam, lr 1e-3, batch 64,
20 epochs, final-epoch weights, seeds 0-2, no hyperparameter tuning (paper defaults).

## Results

**Table 3. Accuracy (%), mean +- standard deviation over 3 seeds (rho = 0.99)**

| Method | ID | OOD-1 | OOD-2 | Gap ID-OOD1 | Gap ID-OOD2 | Worst group (OOD-1) | Colour labels used |
|---|---|---|---|---|---|---|---|
| Baseline | 99.42 +- 0.05 | 92.35 +- 0.70 | 86.93 +- 0.98 | 7.07 +- 0.72 | 12.48 +- 1.00 | 82.66 +- 1.44 | - |
| Reweighting | 98.95 +- 0.31 | **97.96 +- 0.42** | **97.26 +- 0.61** | **0.99 +- 0.46** | **1.69 +- 0.73** | **96.16 +- 0.89** | yes |
| Group DRO | **99.44 +- 0.13** | 96.29 +- 0.39 | 93.80 +- 0.75 | 3.15 +- 0.35 | 5.64 +- 0.67 | 91.89 +- 0.49 | yes |
| LfF | 98.94 +- 0.13 | 95.63 +- 0.22 | 93.02 +- 0.38 | 3.31 +- 0.35 | 5.92 +- 0.52 | 90.46 +- 1.83 | **no** |

**Change versus the baseline with the same seed (points)**

| Method | ID | OOD-1 | OOD-2 |
|---|---|---|---|
| Reweighting | -0.47 | +5.61 | +10.33 (+10.99, +11.17, +8.82) |
| Group DRO | +0.02 | +3.94 | +6.87 (+6.17, +7.75, +6.68) |
| LfF | -0.47 (-0.41, -0.44, -0.57) | +3.28 (+3.15, +3.90, +2.80) | +6.09 (+5.75, +6.98, +5.53) |

![Baseline vs mitigation methods, rho = 0.99](figures/table3_comparison_rho0.99.png)

*Figure 1. Accuracy per method; bars are means, dots individual seeds (y-axis starts at 79%).*

**Auxiliary biased model of LfF (seed 0 only).** ID 99.26, OOD-1 91.63, OOD-2 85.79 (baseline: 99.47,
92.30, 86.94); loss on OOD-2 2.94 against 0.77 for the baseline.

## Findings

1. **LfF improves robustness in every seed without any colour label:** OOD-2 +6.1 points (5.5-7.0),
   OOD-1 +3.3, worst group +7.8 (82.7 to 90.5), ID-OOD2 gap from 12.5 to 5.9 points.
2. **It is weaker than reweighting** (OOD-2 93.0% vs 97.3%) and **close to Group DRO** (93.0% vs 93.8%).
   Group DRO is ahead in each seed, by 0.4 to 1.2 points, but the seed ranges overlap, so with three
   seeds the two cannot be separated reliably.
3. **LfF pays an ID cost** (-0.47 points in every seed), the same as reweighting and unlike Group DRO.
4. **LfF recovers about 89% of Group DRO's and 59% of reweighting's OOD-2 gain** while using strictly less
   information (no bias attribute).
5. **Worst-group accuracy is variable for LfF** (89.5, 89.3 and 92.6 by seed; std 1.83).
6. **The helper model is more overconfident than accurate:** its OOD-2 accuracy is about that of the
   baseline (85.8%, within the baseline's seed spread), but its loss is almost four times higher.

## Interpretation

Methods that use the bias label (reweighting, Group DRO) can target the conflicting samples directly.
LfF has to infer them from the failures of a biased helper, and it recovers most, though not all, of the
benefit. This follows the expected ordering: the more supervision about the bias, the larger the gain.
All three methods narrow the ID-OOD gap, and reweighting nearly closes it.

## Caveats

- Three seeds; same-seed repeated runs also vary (Step 8), so differences of about one point should not be ranked.
- Default hyperparameters for all methods; none tuned.
- The claim that LfF up-weights conflicting samples has not been verified here: the per-epoch mean weights
  W on aligned and conflicting samples are logged but not yet analysed. The accuracy gains are consistent
  with it but do not prove it.
- The auxiliary-model numbers come from one seed.
