"""Compare misspecification results of several baselines (needs only matplotlib).

    python -m plot_misspecification --rhos 0.9 0.99 --seed 0

Output: reports/figures/misspec_comparison_seed{S}.png
Top row: accuracy. Bottom row: how often the predicted group (0-4 / 5-9) follows the colour.
Whiskers = 95% binomial CI. The y-axis starts at 0.
"""
import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SHAPES = ["clean", "rot90", "occlude_right_half"]
SHAPE_LABELS = ["clean", "rotated 90\u00b0", "right half hidden"]
COLOURS = ["natural", "reversed", "grayscale"]
COLOUR_STYLE = {"natural": "#1f77b4", "reversed": "#ff7f0e", "grayscale": "#7f7f7f"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.9, 0.99])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default="./outputs/results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"./reports/figures/misspec_comparison_seed{a.seed}.png"

    fig, ax = plt.subplots(2, len(a.rhos), figsize=(5.4 * len(a.rhos), 7.4), squeeze=False)
    for j, rho in enumerate(a.rhos):
        tag = "" if abs(rho - 0.9) < 1e-9 else f"_rho{rho:g}"
        with open(os.path.join(a.results_dir, f"misspec{tag}_seed{a.seed}.json")) as f:
            res = json.load(f)
        n = res["n"]
        for r, (metric, cols, title, ref, ref_label) in enumerate([
                ("acc", COLOURS, "accuracy (%)", 10, "random guessing"),
                ("follows_colour", COLOURS[:2], "prediction follows the colour (%)", 50, "colour has no pull")]):
            w = 0.8 / len(cols)
            for k, col in enumerate(cols):
                vals = [100 * res[f"{s}/{col}"][metric] for s in SHAPES]
                errs = [196 * math.sqrt(p / 100 * (1 - p / 100) / n) for p in vals]
                xs = [i + (k - (len(cols) - 1) / 2) * w for i in range(len(SHAPES))]
                bars = ax[r][j].bar(xs, vals, w, yerr=errs, capsize=3, color=COLOUR_STYLE[col], label=col)
                for b, v in zip(bars, vals):
                    ax[r][j].text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.0f}", ha="center", fontsize=7)
            ax[r][j].axhline(ref, color="gray", linestyle="--", linewidth=1)
            ax[r][j].text(len(SHAPES) - 0.55, ref + 1.5, ref_label, fontsize=7, color="black", ha="right",
                           bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1))
            ax[r][j].set_xticks(range(len(SHAPES)))
            ax[r][j].set_xticklabels(SHAPE_LABELS, fontsize=8)
            ax[r][j].set_ylim(0, 112)
            ax[r][j].set_ylabel(title)
            ax[r][j].grid(alpha=0.3, axis="y")
            ax[r][j].set_title(f"trained with rho = {rho:g}: {title}", fontsize=9)
            ax[r][j].legend(fontsize=7, loc="upper right", title="colour shown", title_fontsize=7)
    fig.suptitle(f"Misspecification tests (n = {n} test digits, seed {a.seed})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()