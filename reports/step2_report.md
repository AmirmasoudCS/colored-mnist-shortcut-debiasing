# Step 2 Report: Baseline CNN (Task 1)

**Date:** 24 October 2026

## Setup

SimpleCNN (3 conv layers + 2 fully connected, 390,986 parameters) trained on Colored MNIST
(rho = 0.9, 50,000 samples). Adam, lr = 1e-3, batch size 64, 20 epochs, cross-entropy loss.
Single run, seed 0. Weights are taken from the final epoch.

## Results

| | Loss | Accuracy |
|---|---|---|
| Train (epoch 20) | 0.0011 | 99.97% |
| Validation (epoch 20) | 0.0310 | 99.33% |

Validation accuracy was already 97.71% after epoch 1 and plateaued at about 99.2-99.4% from epoch 7.

![Training and validation loss and accuracy over 20 epochs](figures/baseline_curves_seed0.png)

*Figure 1. Loss and accuracy per epoch (train and validation). Dashed lines show random guessing.*

## Observations

- **Task requirement met:** validation accuracy (99.33%) is far above the 80% threshold.
- **Mild overfitting:** train loss keeps decreasing (to 0.001) while validation loss stays between
  0.026 and 0.043 after epoch 9, with no loss of validation accuracy.
- **Shape is used, not only colour:** a colour-only classifier can reach at most about 90% on a
  validation set with rho = 0.9. The model's 99.33% implies it classifies at least about 93% of
  the conflicting (colour-opposite) samples correctly, so it must rely on shape for them.
- **Consequence for the handout's hypothesis:** the predicted collapse on OOD-2 is therefore
  unlikely to appear. This is a bound inferred from validation data and still has to be confirmed
  on the OOD sets (Task 2).
- **Caveat:** the final epoch (99.33%) is reported, not the best validation epoch (epoch 16,
  99.45%), to avoid selecting on validation data.
