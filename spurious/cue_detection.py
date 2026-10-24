"""Task 3: which cue does the model use?  (3.1: counterfactual tests)

All transforms take a float batch x (B, 3, 28, 28) in [0, 1] plus labels y and colours c,
and return a new batch. Colour is a whole channel (0 = red, 1 = green), so the digit's
shape is recovered as x.max over channels.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from .evaluation import evaluate  # noqa: E402


def set_color(x, c):
    """Re-paint every digit in colour c[i] (0 red, 1 green), keeping its shape."""
    gray = x.max(dim=1).values                       # (B, 28, 28)
    out = torch.zeros_like(x)
    out[torch.arange(x.shape[0], device=x.device), c] = gray
    return out


def natural_color(x, y, c):
    """Reference: 0-4 red, 5-9 green (every sample aligned)."""
    return set_color(x, (y >= 5).long())


def anti_color(x, y, c):
    """Test A: 0-4 green, 5-9 red (every sample conflicting)."""
    return set_color(x, 1 - (y >= 5).long())


def grayscale_mean(x, y, c):
    """Test B (handout): all three channels = mean of the channels.
    NOTE: this also divides the digit's intensity by 3 (a pure red digit becomes 1/3 grey)."""
    return x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)


def grayscale_max(x, y, c):
    """Control for Test B: colour removed but full intensity kept (max over channels)."""
    return x.max(dim=1, keepdim=True).values.repeat(1, 3, 1, 1)


CONDITIONS = {
    "clean": None,
    "natural_color": natural_color,        # all samples aligned
    "A_anti_color": anti_color,            # all samples conflicting
    "B_grayscale_mean": grayscale_mean,    # handout's Test B
    "B2_grayscale_max": grayscale_max,     # Test B with intensity preserved
}

CONDITION_LABELS = {
    "clean": "clean test set",
    "natural_color": "all aligned (reference)",
    "A_anti_color": "Test A: colours reversed",
    "B_grayscale_mean": "Test B: grayscale (mean)",
    "B2_grayscale_max": "Test B': grayscale (max, control)",
}


def counterfactual_tests(model, split, device):
    """Accuracy and loss of `model` on `split` under each condition.

    Because every transform overwrites the colour, the transformed results depend only
    on the digit shapes, not on the split's original colouring.
    colour_swing = acc(all aligned) - acc(all conflicting): 0 for a pure shape model.
    """
    res = {"n": int(split[1].shape[0])}
    for name, tf in CONDITIONS.items():
        loss, acc = evaluate(model, split, device, transform=tf)
        res[name] = {"acc": acc, "loss": loss}
    res["colour_swing"] = res["natural_color"]["acc"] - res["A_anti_color"]["acc"]
    return res


def plot_examples(split, path, n=8):
    """Visual check of the transforms (no model needed): original, A, B, B'."""
    x = split[0][:n].float().div(255.0)
    y, c = split[1][:n], split[2][:n]
    rows = [("original", x), ("A: reversed", anti_color(x, y, c)),
            ("B: gray (mean)", grayscale_mean(x, y, c)), ("B': gray (max)", grayscale_max(x, y, c))]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.3 * n + 1.6, 1.3 * len(rows) + 0.6), squeeze=False)
    for r, (label, batch) in enumerate(rows):
        for i in range(n):
            ax = axes[r][i]
            ax.imshow(batch[i].permute(1, 2, 0).clamp(0, 1).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"label {int(y[i])}", fontsize=8)
        axes[r][0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)