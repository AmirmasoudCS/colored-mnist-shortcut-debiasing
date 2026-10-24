"""Step 1 sanity checks. Run from the repo root:  python step1_check_data.py
Every check prints something you can verify by eye."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from spurious.data import build_datasets, colorize, load_mnist  # noqa: E402

FIG_DIR = "reports/figures"  # figures used by the step reports live here


def natural_fraction(y, c):
    """Fraction of samples whose colour equals the natural colour of their digit."""
    return (c == (y >= 5).long()).float().mean().item()


def main():
    # --- Piece 1: loading -------------------------------------------------
    x_tr, y_tr, x_te, y_te = load_mnist("./data")
    print("train:", tuple(x_tr.shape), x_tr.dtype, "| test:", tuple(x_te.shape), x_te.dtype)
    # expected: (60000, 28, 28) uint8 and (10000, 28, 28) uint8

    # --- Piece 3a: fraction of 'natural' colours ~= rho --------------------
    print("\nrho  -> measured fraction of natural colours")
    for rho in (0.9, 0.5, 0.1):
        x, y, c = colorize(x_te, y_te, rho, seed=0)
        print(f"{rho:.1f} -> {natural_fraction(y, c):.4f}")

    # --- Piece 3b: exactly one non-zero channel per image ------------------
    x, y, c = colorize(x_te, y_te, 0.9, seed=0)
    nonzero_channels = (x.flatten(2).amax(dim=2) > 0).sum(dim=1)  # (N,)
    assert (nonzero_channels == 1).all(), "some image has 0 or >1 coloured channels"
    print("\nevery image has exactly one non-zero channel: OK")
    # the coloured channel must match c (0 = red, 1 = green)
    active = x.flatten(2).amax(dim=2).argmax(dim=1)
    assert torch.equal(active, c), "active channel does not match c"
    print("active channel == c for every image: OK")

    # --- Piece 3c: reproducibility -----------------------------------------
    _, _, c_a = colorize(x_te, y_te, 0.9, seed=0)
    _, _, c_b = colorize(x_te, y_te, 0.9, seed=0)
    _, _, c_c = colorize(x_te, y_te, 0.9, seed=1)
    print("\nsame seed -> identical colours:", torch.equal(c_a, c_b))
    print("different seed -> different colours:", not torch.equal(c_a, c_c))

    # --- Piece 3d: look at it ----------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i].permute(1, 2, 0).numpy())  # (3,28,28) -> (28,28,3), uint8
        ax.set_title(f"label {int(y[i])}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "check_samples.png")
    fig.savefig(out_path, dpi=130)
    print(f"\nsaved {out_path}")

    # --- Piece 4: the five splits ------------------------------------------
    data = build_datasets("./data")
    print("\nset        size   rho(expected)  natural fraction")
    expected = {"train": 0.9, "val": 0.9, "test_id": 0.9, "test_ood1": 0.5, "test_ood2": 0.1}
    for name, (xs, ys, cs) in data.items():
        print(f"{name:<10} {len(ys):>6}   {expected[name]:.1f}            {natural_fraction(ys, cs):.4f}")

    # class balance of the training set (should be roughly 10% per digit)
    counts = torch.bincount(data["train"][1], minlength=10)
    print("\ntrain digits per class:", counts.tolist())


if __name__ == "__main__":
    main()