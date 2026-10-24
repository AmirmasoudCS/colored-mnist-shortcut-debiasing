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
    """
    train = datasets.MNIST(root, train=True, download=True)
    test = datasets.MNIST(root, train=False, download=True)
    return train.data, train.targets, test.data, test.targets


def colorize(gray, y, rho, seed):
    """Colour grayscale digits.

    gray: uint8 (N, 28, 28), y: int64 (N,)
    Returns: x uint8 (N, 3, 28, 28), y (N,), c int64 (N,)  with c in {0, 1}
    """
    gray, y = gray.cpu(), y.cpu()
    n = gray.shape[0]

    # 1. natural colour of every sample: 0-4 -> 0 (red), 5-9 -> 1 (green)
    natural = (y >= 5).long()

    # 2. keep the natural colour with probability rho (seeded, reproducible)
    gen = torch.Generator().manual_seed(seed)
    keep = torch.rand(n, generator=gen) < rho

    # 3. final colour: natural where keep, the opposite colour elsewhere
    c = torch.where(keep, natural, 1 - natural)

    # 4. paste each digit into exactly one channel (0 = red, 1 = green)
    x = torch.zeros(n, 3, 28, 28, dtype=torch.uint8)
    for color in (0, 1):
        mask = c == color
        x[mask, color] = gray[mask]

    return x, y.clone(), c


def build_datasets(root="./data", seed=DATA_SEED):
    """Return a dict of (x, y, c) tuples with keys:
    'train' (50k, rho .9), 'val' (10k, rho .9),
    'test_id' (10k, rho .9), 'test_ood1' (10k, rho .5), 'test_ood2' (10k, rho .1)

    Train/val: the 60k MNIST *training* digits are split 50k/10k with a seeded
    randperm, so the two sets contain different digits.

    Test sets: MNIST has only 10,000 test digits, so the handout's slicing
    (test_indices[10000:20000], [20000:30000]) would give EMPTY sets. We re-use the
    same 10,000 held-out test digits three times with different colourings
    (rho = 0.9 / 0.5 / 0.1). They never overlap with train/val digits, so there is
    no data leakage; the three test sets simply share their shapes.
    """
    x_tr, y_tr, x_te, y_te = load_mnist(root)

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x_tr.shape[0], generator=gen)
    tr_idx, val_idx = perm[:50000], perm[50000:]

    return {
        "train": colorize(x_tr[tr_idx], y_tr[tr_idx], 0.9, seed),
        "val": colorize(x_tr[val_idx], y_tr[val_idx], 0.9, seed + 1),
        "test_id": colorize(x_te, y_te, 0.9, seed + 2),
        "test_ood1": colorize(x_te, y_te, 0.5, seed + 3),
        "test_ood2": colorize(x_te, y_te, 0.1, seed + 4),
    }