import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CHANCE_LOSS = math.log(10)  # cross-entropy of uniform guessing over 10 classes (~2.303)
LOSS_TOP = 2.5              # default upper limit of the loss axis (extended if needed)


def plot_history(hist, path, title):
    """Loss and accuracy curves, train vs validation, on fixed axes:
    loss in [0, max(2.5, data)], accuracy in [0, 100] %."""
    n = len(hist["train_loss"])
    epochs = list(range(1, n + 1))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))

    # ---- loss ----
    ax[0].plot(epochs, hist["train_loss"], label="train")
    ax[0].plot(epochs, hist["val_loss"], label="validation")
    ax[0].axhline(CHANCE_LOSS, color="gray", linestyle="--", linewidth=1,
                  label="random guessing (ln 10)")
    top = max(LOSS_TOP, 1.05 * max(hist["train_loss"] + hist["val_loss"]))
    ax[0].set_ylim(0, top)
    ax[0].set_ylabel("cross-entropy loss")
    ax[0].set_title("Loss")

    # ---- accuracy (shown in %) ----
    ax[1].plot(epochs, [100 * a for a in hist["train_acc"]], label="train")
    ax[1].plot(epochs, [100 * a for a in hist["val_acc"]], label="validation")
    ax[1].axhline(10, color="gray", linestyle="--", linewidth=1, label="random guessing (10%)")
    ax[1].set_ylim(0, 100)
    ax[1].set_ylabel("accuracy (%)")
    ax[1].set_title("Accuracy")

    for a in ax:
        a.set_xlabel("epoch")
        a.set_xlim(1, max(n, 2))
        if n <= 25:
            a.set_xticks(epochs)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)