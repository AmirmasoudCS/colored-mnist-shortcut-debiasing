"""Step 1 sanity checks. Run from the repo root:  python step1_check_data.py
Fill in the TODOs. Every check should print something you can verify by eye."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from spurious.data import colorize, load_mnist  # noqa: E402


def main():
    # --- Piece 1: loading -------------------------------------------------
    x_tr, y_tr, x_te, y_te = load_mnist("./data")
    print("train:", x_tr.shape, x_tr.dtype, "| test:", x_te.shape, x_te.dtype)
    # expected: (60000, 28, 28) uint8 and (10000, 28, 28) uint8

    # --- Piece 3a: fraction of 'natural' colours ~= rho --------------------
    for rho in (0.9, 0.5, 0.1):
        x, y, c = colorize(x_te, y_te, rho, seed=0)
        # TODO: compute the fraction of samples whose colour equals the natural one
        #       and print it next to rho
        pass

    # --- Piece 3b: exactly one non-zero channel per image ------------------
    x, y, c = colorize(x_te, y_te, 0.9, seed=0)
    # TODO: for each image, how many channels contain any non-zero pixel?
    #       Assert it is exactly 1 for every image.

    # --- Piece 3c: reproducibility -----------------------------------------
    # TODO: same seed twice -> identical c ? different seed -> different c ?

    # --- Piece 3d: look at it ----------------------------------------------
    # TODO: plot a 2x5 grid of x[:10] (use .permute(1, 2, 0) for imshow),
    #       title each with its label, save to 'check_samples.png'


if __name__ == "__main__":
    main()