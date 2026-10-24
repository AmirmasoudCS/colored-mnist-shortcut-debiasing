"""Compare the counterfactual results of several baselines (needs only matplotlib).

    python -m plot_counterfactual --rhos 0.9 0.99 --seed 0

Output: reports/figures/counterfactual_comparison_seed{S}.png
Bars: accuracy (%) per condition, one colour per training rho; whiskers = 95% binomial CI
(n = 10,000 test digits). The y-axis starts at 0.
"""
import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONDITIONS = ["clean", "natural_color", "A_anti_color", "B_grayscale_mean", "B2_grayscale_max"]
LABELS = ["clean", "all aligned\n(reference)", "Test A:\ncolours reversed",
          "Test B:\ngrayscale (mean)", "Test B':\ngrayscale (max)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.9, 0.99])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default="./outputs/results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"./reports/figures/counterfactual_comparison_seed{a.seed}.png"

    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.8 / len(a.rhos)
    for j, rho in enumerate(a.rhos):
        tag = "" if abs(rho - 0.9) < 1e-9 else f"_rho{rho:g}"
        with open(os.path.join(a.results_dir, f"counterfactual{tag}_seed{a.seed}.json")) as f:
            res = json.load(f)
        vals = [100 * res[c]["acc"] for c in CONDITIONS]
        errs = [196 * math.sqrt(res[c]["acc"] * (1 - res[c]["acc"]) / res["n"]) for c in CONDITIONS]
        xs = [i + (j - (len(a.rhos) - 1) / 2) * w for i in range(len(CONDITIONS))]
        bars = ax.bar(xs, vals, w, yerr=errs, capsize=3, label=f"trained with rho = {rho:g}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 108)
    ax.axhline(10, color="gray", linestyle="--", linewidth=1)
    ax.text(len(CONDITIONS) - 0.5, 11.5, "random guessing", fontsize=7, color="gray", ha="right")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(f"Counterfactual tests on the 10,000 test digits (seed {a.seed})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()