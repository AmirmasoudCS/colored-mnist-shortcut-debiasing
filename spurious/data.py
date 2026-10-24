"""Colored MNIST construction  (Step 1).

Convention used everywhere in the project:
    colour c: 0 = red, 1 = green
    "natural" pairing: digits 0-4 -> red (c = 0), digits 5-9 -> green (c = 1)
    with probability rho a sample gets its natural colour, otherwise the opposite one.
"""
import torch
from torchvision import datasets

DATA_SEED = 42  # the dataset is fixed; only model seeds will change later


def load_mnist(root="./data"):
    """Return (x_train, y_train, x_test, y_test).

    x_*: uint8 tensors (N, 28, 28);  y_*: int64 tensors (N,)
    Hint: datasets.MNIST(root, train=True/False, download=True) and its
          .data / .targets attributes. No ToTensor() and no loop needed.
    """
    # TODO
    raise NotImplementedError


def colorize(gray, y, rho, seed):
    """Colour grayscale digits.

    gray: uint8 (N, 28, 28), y: int64 (N,)
    Returns: x uint8 (N, 3, 28, 28), y (N,), c int64 (N,)  with c in {0, 1}

    Plan:
      1. natural colour per sample from y              (no loops)
      2. keep = rand(N) < rho, using a seeded torch.Generator
      3. c = natural where keep, else the opposite colour
      4. x = zeros(N, 3, 28, 28, dtype=torch.uint8); paste each digit into
         channel 0 (red) or channel 1 (green) with boolean masks
    """
    # TODO
    raise NotImplementedError


def build_datasets(root="./data", seed=DATA_SEED):
    """Return a dict of (x, y, c) tuples with keys:
    'train' (50k, rho .9), 'val' (10k, rho .9),
    'test_id' (10k, rho .9), 'test_ood1' (10k, rho .5), 'test_ood2' (10k, rho .1)

    Train/val: split the 60k MNIST *training* digits with a seeded randperm.
    Test sets: decide how to handle the handout's bug (MNIST has only 10k test
    digits) -- we agreed to discuss the plan before coding this part.
    Use a different colouring seed for every set.
    """
    # TODO (do this last, after colorize passes the checks)
    raise NotImplementedError