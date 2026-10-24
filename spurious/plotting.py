import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_history(hist, path, title):
    """Loss and accuracy curves, train vs validation."""
    epochs = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))

    ax[0].plot(epochs, hist["train_loss"], label="train")
    ax[0].plot(epochs, hist["val_loss"], label="validation")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("cross-entropy loss")
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(epochs, hist["train_acc"], label="train")
    ax[1].plot(epochs, hist["val_acc"], label="validation")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].set_title("Accuracy")
    ax[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)